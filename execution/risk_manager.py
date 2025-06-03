# execution/risk_manager.py
"""
Manages trade risk, including Stop Loss (SL) and Take Profit (TP) calculation,
and flexible position sizing (fixed lot or dynamic percentage of equity).
"""
import numpy as np
import math
import pandas as pd # For pd.isna checks

# Assuming config and logging_utils are accessible
# (Path adjustments might be needed if run standalone)
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in RiskManager. Ensure paths are correct.")
    # Fallback for basic logging if utilities are not found during direct execution
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # A mock config might be needed for standalone testing if config is not found
    class MockConfig:
        RISK_MANAGEMENT_MODE = "DYNAMIC_PERCENTAGE"
        FIXED_LOT_SIZE = 0.01
        RISK_PER_TRADE_PERCENT = 1.0
        MIN_DYNAMIC_LOT_SIZE = 0.01
        MAX_DYNAMIC_LOT_SIZE = 2.0
        DEFAULT_RRR_EXECUTION = 1.5
        SL_ATR_MULTIPLIER_EXECUTION = 1.5
        PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}
        HISTORICAL_DATA_SOURCES = {
            "EURUSD_M5": {
                "pair": "EURUSD", "pip_size": 0.0001,
                "backtest_mt5_point": 0.00001, "backtest_mt5_trade_tick_value": 1.0,
                "backtest_mt5_volume_min": 0.01, "backtest_mt5_volume_max": 100.0,
                "backtest_mt5_volume_step": 0.01, "backtest_account_currency": "USD"
            }
        }
        # Add other necessary mock config attributes if testing specific functions
    if 'config' not in locals() and 'config' not in globals(): # Check if config was truly not imported
        config = MockConfig()
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP if hasattr(config, 'LOG_FILE_APP') else "risk_manager_temp.log")

else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


class RiskManager:
    def __init__(self, config_obj, logger_obj): # Changed to accept config and logger
        self.config = config_obj
        self.logger = logger_obj

        self.risk_mode = getattr(self.config, 'RISK_MANAGEMENT_MODE', "DYNAMIC_PERCENTAGE")
        self.fixed_lot = getattr(self.config, 'FIXED_LOT_SIZE', 0.01)
        self.risk_percent = getattr(self.config, 'RISK_PER_TRADE_PERCENT', 1.0) / 100.0 # Convert to decimal
        self.min_dyn_lot = getattr(self.config, 'MIN_DYNAMIC_LOT_SIZE', 0.01)
        self.max_dyn_lot = getattr(self.config, 'MAX_DYNAMIC_LOT_SIZE', 2.0)

        self.logger.info(f"RiskManager initialized. Mode: {self.risk_mode}, FixedLot: {self.fixed_lot}, Risk%: {self.risk_percent*100:.2f}%")

    def _get_symbol_pip_size(self, symbol: str) -> float:
        """ Helper to get pip size from config.PIP_SIZE for a given symbol. """
        default_pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
        pip_size = self.config.PIP_SIZE.get(symbol, default_pip_size)
        if pip_size <= 0:
            self.logger.warning(f"Configured pip size for {symbol} is {pip_size}. Using emergency default {default_pip_size}.")
            return default_pip_size
        return pip_size

    def _get_digits_for_rounding(self, value_with_precision: float) -> int:
        """ Determines decimal places for rounding based on a value like pip_size or lot_step. """
        if value_with_precision <= 0:
            self.logger.warning(f"Cannot determine digits for rounding from non-positive value: {value_with_precision}. Defaulting to 5 digits.")
            return 5
        # Convert to string, split by decimal, take length of fractional part
        s_val = f"{value_with_precision:.10f}" # Format to ensure enough decimal places
        if '.' in s_val:
            return len(s_val.split('.')[1].rstrip('0'))
        return 0 # For whole numbers

    def calculate_stop_loss(self, entry_price: float, atr_at_entry: float, direction: str, symbol: str) -> float | None:
        direction_lower = direction.lower()
        if pd.isna(entry_price) or pd.isna(atr_at_entry) or atr_at_entry <= 0:
            self.logger.warning(f"Cannot calculate SL for {symbol} due to invalid entry_price ({entry_price}) or atr_at_entry ({atr_at_entry}).")
            return None

        sl_atr_multiplier = getattr(self.config, 'SL_ATR_MULTIPLIER_EXECUTION', 1.5)
        pip_size = self._get_symbol_pip_size(symbol)
        digits_to_round = self._get_digits_for_rounding(pip_size) # Round SL to pip precision

        sl_distance_points = sl_atr_multiplier * atr_at_entry

        if direction_lower in ['buy', 'long']:
            sl_price = entry_price - sl_distance_points
        elif direction_lower in ['sell', 'short']:
            sl_price = entry_price + sl_distance_points
        else:
            self.logger.error(f"Invalid direction for SL: '{direction}'. Must be 'buy'/'long' or 'sell'/'short'.")
            return None

        sl_price_rounded = round(sl_price, digits_to_round)
        self.logger.debug(f"SL Calc for {symbol} ({direction_lower}): Entry={entry_price:.{digits_to_round}f}, ATR={atr_at_entry:.5f}, SL_Dist_Pts={sl_distance_points:.{digits_to_round}f}, SL_Raw={sl_price:.{digits_to_round}f}, SL_Rounded={sl_price_rounded:.{digits_to_round}f}")
        return sl_price_rounded

    def calculate_take_profit(self, entry_price: float, sl_price: float, direction: str, symbol: str, rrr: float = None) -> float | None:
        direction_lower = direction.lower()
        if sl_price is None or pd.isna(entry_price) or pd.isna(sl_price):
            self.logger.warning(f"SL price is None/NaN for {symbol} (Entry: {entry_price}), cannot calculate TP.")
            return None

        rrr_to_use = rrr if rrr is not None and rrr > 0 else getattr(self.config, 'DEFAULT_RRR_EXECUTION', 1.5)
        if rrr_to_use <= 0:
            self.logger.warning(f"RRR for TP calculation is {rrr_to_use:.2f} (invalid). Cannot calculate TP.")
            return None

        pip_size = self._get_symbol_pip_size(symbol)
        digits_to_round = self._get_digits_for_rounding(pip_size)
        sl_distance_abs_points = abs(entry_price - sl_price)

        if direction_lower in ['buy', 'long']:
            tp_price = entry_price + rrr_to_use * sl_distance_abs_points
        elif direction_lower in ['sell', 'short']:
            tp_price = entry_price - rrr_to_use * sl_distance_abs_points
        else:
            self.logger.error(f"Invalid direction for TP: '{direction}'.")
            return None

        tp_price_rounded = round(tp_price, digits_to_round)
        self.logger.debug(f"TP Calc for {symbol} ({direction_lower}): Entry={entry_price:.{digits_to_round}f}, SL={sl_price:.{digits_to_round}f}, RRR={rrr_to_use:.2f}, TP_Raw={tp_price:.{digits_to_round}f}, TP_Rounded={tp_price_rounded:.{digits_to_round}f}")
        return tp_price_rounded

    def _get_symbol_params_for_sizing(self, symbol: str, symbol_info_mt5=None):
        """Helper to get point, tick_value, vol_min, vol_max, vol_step for sizing."""
        if symbol_info_mt5: # Live trading
            return {
                "point": symbol_info_mt5.point,
                "trade_tick_value": symbol_info_mt5.trade_tick_value, # Value of one point move for 1 lot
                "volume_min": symbol_info_mt5.volume_min,
                "volume_max": symbol_info_mt5.volume_max,
                "volume_step": symbol_info_mt5.volume_step,
                "currency": symbol_info_mt5.currency_profit # Or currency_base, check MT5 docs
            }
        else: # Backtesting - get from config
            # Derive m5_pair_key (e.g., "EURUSD_M5") from symbol ("EURUSD")
            # This assumes symbol is just the pair name like "EURUSD"
            m5_pair_key = f"{symbol.upper()}_{self.config.TIMEFRAME_M5_STR}" # e.g. EURUSD_M5
            pair_data_config = self.config.HISTORICAL_DATA_SOURCES.get(m5_pair_key)
            if pair_data_config:
                return {
                    "point": pair_data_config.get("backtest_mt5_point"),
                    "trade_tick_value": pair_data_config.get("backtest_mt5_trade_tick_value"),
                    "volume_min": pair_data_config.get("backtest_mt5_volume_min"),
                    "volume_max": pair_data_config.get("backtest_mt5_volume_max"),
                    "volume_step": pair_data_config.get("backtest_mt5_volume_step"),
                    "currency": pair_data_config.get("backtest_account_currency")
                }
        self.logger.error(f"Could not retrieve sizing parameters for symbol {symbol} (live or backtest).")
        return None

    def _calculate_position_size_dynamic(self, account_balance: float, entry_price: float, sl_price: float,
                                         symbol: str, symbol_info_mt5=None) -> float | None:
        """ Calculates dynamic position size based on risk percentage. """
        if sl_price is None or entry_price == sl_price or pd.isna(entry_price) or pd.isna(sl_price):
            self.logger.error(f"Invalid SL ({sl_price}) or SL at entry ({entry_price}) for {symbol}. Cannot size dynamically.")
            return None
        if account_balance <= 0:
            self.logger.error(f"Invalid account balance: {account_balance:.2f}. Cannot size dynamically.")
            return None

        sizing_params = self._get_symbol_params_for_sizing(symbol, symbol_info_mt5)
        if not sizing_params or not all(k in sizing_params and sizing_params[k] is not None for k in ["point", "trade_tick_value", "volume_min", "volume_step"]):
            self.logger.error(f"Missing critical sizing parameters for {symbol}. Cannot size dynamically."); return None
        
        point_size = sizing_params["point"]
        tick_value_per_lot = sizing_params["trade_tick_value"] # Value of one 'point' move for 1 lot
        min_lot = sizing_params["volume_min"]
        max_lot = sizing_params.get("volume_max", 1000.0) # Default if not specified
        lot_step = sizing_params["volume_step"]

        if point_size <= 0 or tick_value_per_lot <= 0 or min_lot <=0 or lot_step <=0:
            self.logger.error(f"Invalid sizing params for {symbol}: point={point_size}, tick_val={tick_value_per_lot}, min_lot={min_lot}, lot_step={lot_step}. Cannot size."); return None

        sl_distance_points_abs = abs(entry_price - sl_price) # SL distance in price units (e.g., 0.00500)
        sl_distance_in_min_increments = sl_distance_points_abs / point_size # SL distance in 'points' or 'ticks'
        
        if sl_distance_in_min_increments <= 0:
            self.logger.error(f"SL distance in points for {symbol} is zero/negative ({sl_distance_in_min_increments:.2f}). Cannot size.")
            return None

        risk_amount_currency = account_balance * self.risk_percent
        value_of_sl_per_lot = sl_distance_in_min_increments * tick_value_per_lot # Total risk in currency for 1 lot

        if value_of_sl_per_lot <= 0:
            self.logger.error(f"Value of SL per lot for {symbol} is {value_of_sl_per_lot:.5f}. Cannot size.")
            return None

        position_size_lots_raw = risk_amount_currency / value_of_sl_per_lot
        
        self.logger.info(f"Dynamic Position Sizing for {symbol}:")
        self.logger.info(f"  Account Balance: {account_balance:.2f}, Risk %: {self.risk_percent*100:.2f}%")
        self.logger.info(f"  Risk Amount (currency): {risk_amount_currency:.2f}")
        self.logger.info(f"  SL Distance (price units): {sl_distance_points_abs:.5f}")
        self.logger.info(f"  SL Distance (points/ticks): {sl_distance_in_min_increments:.1f}")
        self.logger.info(f"  Tick Value (1 lot, currency): {tick_value_per_lot:.2f} per point")
        self.logger.info(f"  Value of SL (1 lot, currency): {value_of_sl_per_lot:.2f}")
        self.logger.info(f"  Raw Position Size (lots): {position_size_lots_raw:.5f}")

        # Apply constraints: min_lot, max_lot (config), lot_step
        constrained_lot_size = max(self.min_dyn_lot, position_size_lots_raw) # Apply config min_dyn_lot
        constrained_lot_size = min(self.max_dyn_lot, constrained_lot_size)   # Apply config max_dyn_lot

        # Adjust for broker's min_lot, max_lot, lot_step
        if constrained_lot_size < min_lot:
            self.logger.warning(f"Calculated lot {constrained_lot_size:.5f} < broker min_lot {min_lot}. Using min_lot.")
            final_lot_size = min_lot
        elif constrained_lot_size > max_lot:
            self.logger.warning(f"Calculated lot {constrained_lot_size:.5f} > broker max_lot {max_lot}. Using max_lot.")
            final_lot_size = max_lot
        else:
            # Adjust for lot step: round down to the nearest valid lot step
            final_lot_size = math.floor(constrained_lot_size / lot_step) * lot_step
            # If rounding down made it less than min_lot (but original was >= min_lot), use min_lot
            if constrained_lot_size >= min_lot and final_lot_size < min_lot:
                final_lot_size = min_lot
        
        # Final check if result is zero after adjustments (e.g. if lot_step is large)
        if final_lot_size <= 0:
            self.logger.error(f"Final lot size for {symbol} is {final_lot_size:.5f} after all adjustments. Cannot place trade.")
            return None
        
        # Round to precision of lot_step
        lot_step_digits = self._get_digits_for_rounding(lot_step)
        final_lot_size_rounded = round(final_lot_size, lot_step_digits)

        self.logger.info(f"  Final Adjusted Position Size (lots) for {symbol}: {final_lot_size_rounded:.{lot_step_digits}f}")
        return final_lot_size_rounded

    def get_trade_volume(self, account_balance: float, entry_price: float, sl_price: float,
                         symbol: str, symbol_info_mt5=None) -> float:
        """ Primary method to get trade volume based on configured risk mode. """
        self.logger.info(f"Determining trade volume for {symbol}. Mode: {self.risk_mode}")
        volume = 0.0

        if self.risk_mode == "FIXED_LOT":
            if self.fixed_lot > 0:
                self.logger.info(f"Using fixed lot size: {self.fixed_lot}")
                volume = self.fixed_lot
            else: # fixed_lot is 0 or invalid, try dynamic as fallback
                self.logger.warning(f"Fixed lot size is {self.fixed_lot}, attempting dynamic percentage sizing as fallback.")
                dyn_volume = self._calculate_position_size_dynamic(account_balance, entry_price, sl_price, symbol, symbol_info_mt5)
                if dyn_volume and dyn_volume > 0:
                    volume = dyn_volume
                else:
                    self.logger.error("Dynamic sizing fallback also failed or resulted in zero volume.")
                    volume = 0.0
        elif self.risk_mode == "DYNAMIC_PERCENTAGE":
            dyn_volume = self._calculate_position_size_dynamic(account_balance, entry_price, sl_price, symbol, symbol_info_mt5)
            if dyn_volume and dyn_volume > 0:
                volume = dyn_volume
            else: # Dynamic failed, try fixed lot as fallback if valid
                self.logger.warning("Dynamic percentage sizing failed or resulted in zero volume.")
                if self.fixed_lot > 0:
                    self.logger.info(f"Falling back to fixed lot size: {self.fixed_lot}")
                    volume = self.fixed_lot
                else:
                    self.logger.error("Fixed lot size is also invalid/zero. Cannot determine volume.")
                    volume = 0.0
        else:
            self.logger.error(f"Invalid RISK_MANAGEMENT_MODE: '{self.risk_mode}'. Defaulting to fixed lot if valid.")
            if self.fixed_lot > 0: volume = self.fixed_lot
            else: volume = 0.0
        
        # Ensure volume is not less than broker's minimum if we have that info
        sizing_params_check = self._get_symbol_params_for_sizing(symbol, symbol_info_mt5)
        if sizing_params_check and sizing_params_check.get("volume_min") is not None:
            broker_min_lot = sizing_params_check["volume_min"]
            if volume > 0 and volume < broker_min_lot:
                self.logger.warning(f"Calculated volume {volume} is below broker minimum {broker_min_lot}. Adjusting to broker minimum.")
                volume = broker_min_lot
            # Also check against broker_max_lot
            broker_max_lot = sizing_params_check.get("volume_max")
            if broker_max_lot is not None and volume > broker_max_lot:
                self.logger.warning(f"Calculated volume {volume} is above broker maximum {broker_max_lot}. Adjusting to broker maximum.")
                volume = broker_max_lot


        if volume <= 0:
            self.logger.error(f"Final trade volume for {symbol} is {volume:.5f}. Trade cannot be placed.")
            return 0.0
        
        # Final rounding based on lot_step if available
        if sizing_params_check and sizing_params_check.get("volume_step") is not None:
            lot_step = sizing_params_check["volume_step"]
            if lot_step > 0:
                lot_step_digits = self._get_digits_for_rounding(lot_step)
                volume = round(math.floor(volume / lot_step) * lot_step, lot_step_digits)


        self.logger.info(f"Final determined trade volume for {symbol}: {volume:.5f}")
        return volume

    # TODO: Implement breakeven and trailing stop logic here if they are to be managed by RiskManager
    # These would typically be called in the live trading loop or backtester's trade management section.
    # def check_and_apply_breakeven(self, position_obj, current_market_price, mt5_interface_obj): ...
    # def check_and_apply_trailing_stop(self, position_obj, current_market_price, current_atr, mt5_interface_obj): ...