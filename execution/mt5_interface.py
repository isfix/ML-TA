# execution/mt5_interface.py
"""
Interface for interacting with the MetaTrader 5 (MT5) trading platform.
Includes functionalities for connecting, fetching data, and executing trades.
Based on Project 1's MT5Interface.
"""
import MetaTrader5 as mt5
import pandas as pd
import time # For retries or delays
import typing # For type hinting, especially for mt5 result objects

# Assuming config and logging_utils are accessible
try:
    import config
    from utilities import logging_utils
except ImportError:
    print("Warning: Could not perform standard config/utilities import in MT5Interface. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Mock config for standalone testing
    class MockConfig:
        LOG_FILE_LIVE = "mt5_interface_temp.log" # Or LOG_FILE_APP
        MT5_LOGIN = "demologin"
        MT5_PASSWORD = "demopassword"
        MT5_SERVER = "demoserver"
        MT5_PATH = ""
        # Add other necessary mock config attributes if testing specific functions
    if 'config' not in locals() and 'config' not in globals():
        config = MockConfig()
        logger = logging_utils.setup_logger(__name__, config.LOG_FILE_LIVE)
else: # Standard import successful
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_LIVE) # Use live log for MT5 interactions


class MT5Interface:
    def __init__(self, config_obj, logger_obj): # Accept config and logger
        self.config = config_obj
        self.logger = logger_obj
        self.is_connected = False
        # Attempt to connect upon initialization if auto_connect is desired,
        # or provide a separate connect() method to be called explicitly.
        # P1's MT5Interface called self.connect() in __init__. Let's keep that.
        self.connect()

    def connect(self) -> bool:
        """Initializes and logs in to the MetaTrader 5 terminal."""
        if self.is_connected and mt5.terminal_info() is not None:
            self.logger.info("Already connected to MT5.")
            return True
            
        self.logger.info(f"Attempting to connect to MT5: Login={self.config.MT5_LOGIN}, Server={self.config.MT5_SERVER}")
        mt5_path_to_use = self.config.MT5_PATH if self.config.MT5_PATH else None

        if not mt5.initialize(login=self.config.MT5_LOGIN,
                              password=self.config.MT5_PASSWORD,
                              server=self.config.MT5_SERVER,
                              path=mt5_path_to_use):
            self.logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}")
            self.is_connected = False
            return False

        self.logger.info(f"MT5 initialized successfully. Version: {mt5.version()}")
        terminal_info = mt5.terminal_info()
        if terminal_info: self.logger.info(f"Terminal: {terminal_info.name}, Build: {terminal_info.build}")

        account_info = mt5.account_info()
        if account_info:
            self.logger.info(f"Logged in to MT5 account: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance:.2f} {account_info.currency}")
            self.is_connected = True
            return True
        else:
            self.logger.error(f"Failed to get account info after MT5 initialize. Error: {mt5.last_error()}. Ensure login details are correct or terminal is logged in.")
            mt5.shutdown() # Shutdown if login effectively failed
            self.is_connected = False
            return False

    def disconnect(self):
        """Shuts down the MetaTrader 5 connection."""
        if mt5.terminal_info(): # Check if there's any active session
            self.logger.info("Shutting down MetaTrader 5 connection.")
            mt5.shutdown()
        else:
            self.logger.info("MetaTrader 5 connection already shut down or was not initialized.")
        self.is_connected = False # Always set to false after attempting disconnect

    def get_account_info(self) -> typing.Optional[mt5.AccountInfo]:
        """Fetches and returns account information."""
        if not self.is_connected and not self.connect(): return None
        info = mt5.account_info()
        if not info: self.logger.error(f"Failed to get account info: {mt5.last_error()}"); return None
        return info

    def get_symbol_info(self, symbol: str) -> typing.Optional[mt5.SymbolInfo]:
        """Fetches and returns information for a specific symbol. Ensures symbol is selected."""
        if not self.is_connected and not self.connect(): return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.warning(f"Symbol {symbol} not found. Attempting to select in MarketWatch.")
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol} in MarketWatch. Error: {mt5.last_error()}")
                return None
            time.sleep(0.2) # Brief pause for MT5
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.error(f"Symbol {symbol} still not found after select. Error: {mt5.last_error()}"); return None
        return info

    def get_symbol_tick_info(self, symbol: str) -> typing.Optional[mt5.SymbolTick]:
        """Fetches and returns the latest tick information (bid/ask) for a symbol."""
        if not self.is_connected and not self.connect(): return None
        # Ensure symbol is available (get_symbol_info handles select if needed)
        if not self.get_symbol_info(symbol):
            self.logger.error(f"Cannot get tick info; symbol details for {symbol} unavailable."); return None

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None: self.logger.error(f"Failed to get tick info for {symbol}: {mt5.last_error()}"); return None
        return tick_info

    def place_market_order(self, symbol: str, order_type_mt5: int, volume: float,
                           sl_price: typing.Optional[float], tp_price: typing.Optional[float],
                           magic_number: int = 0, comment: str = "ai_trade_bot", deviation: int = 10
                           ) -> typing.Optional[mt5.TradeResult]: # TradeResult is the type for order_send
        """Places a market order (BUY or SELL)."""
        if not self.is_connected and not self.connect(): return None

        tick = self.get_symbol_tick_info(symbol)
        if not tick: self.logger.error(f"No tick info for {symbol}. Cannot determine entry price."); return None

        price = tick.ask if order_type_mt5 == mt5.ORDER_TYPE_BUY else tick.bid
        if price == 0: self.logger.error(f"Market price for {symbol} is 0 (Bid: {tick.bid}, Ask: {tick.ask}). Order cannot be placed."); return None

        sl_to_send = float(sl_price) if sl_price is not None and sl_price > 0 else 0.0
        tp_to_send = float(tp_price) if tp_price is not None and tp_price > 0 else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
            "type": order_type_mt5, "price": price, "sl": sl_to_send, "tp": tp_to_send,
            "deviation": deviation, "magic": magic_number, "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Or FOK, check broker requirements
        }
        self.logger.info(f"Sending market order request: {request}")
        
        # Optional: order_check before sending
        # check_result = mt5.order_check(request)
        # if check_result is None or check_result.retcode != mt5.TRADE_RETCODE_DONE:
        #     self.logger.error(f"Order check failed for {symbol}. Retcode: {check_result.retcode if check_result else 'N/A'}, Comment: {check_result.comment if check_result else mt5.last_error()}")
        #     return None # Or return check_result

        order_result = mt5.order_send(request)
        if order_result is None:
            self.logger.error(f"order_send call failed for {symbol}. MT5 error: {mt5.last_error()}"); return None
        
        self.logger.info(f"Order send result for {symbol}: Code={order_result.retcode}, Deal={order_result.deal}, Order={order_result.order}, Comment='{order_result.comment}'")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Order executed successfully: Deal ID {order_result.deal} for {symbol}")
        else:
            self.logger.error(f"Order execution failed for {symbol}: {order_result.comment} (Retcode: {order_result.retcode})")
            self.logger.debug(f"Failed request details: {request}")
        return order_result

    def close_position(self, position_ticket: int, volume_to_close: typing.Optional[float] = None,
                       comment: str = "ai_bot_close") -> typing.Optional[mt5.TradeResult]:
        """Closes an open position by its ticket number. Optionally closes partial volume."""
        if not self.is_connected and not self.connect(): return None

        positions = mt5.positions_get(ticket=position_ticket)
        if not positions: self.logger.error(f"Position with ticket {position_ticket} not found. MT5 Error: {mt5.last_error()}"); return None
        
        position = positions[0]
        symbol = position.symbol
        
        close_volume = float(volume_to_close) if volume_to_close is not None and volume_to_close > 0 else position.volume
        if close_volume <= 0 or close_volume > position.volume:
             self.logger.error(f"Invalid volume {close_volume:.2f} to close for pos {position_ticket} (vol: {position.volume:.2f})"); return None

        tick = self.get_symbol_tick_info(symbol)
        if not tick: self.logger.error(f"No tick info for {symbol} to close pos {position_ticket}."); return None

        close_order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
        if price == 0: self.logger.error(f"Market price for closing {symbol} is 0. Cannot close."); return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": close_volume,
            "type": close_order_type, "position": position.ticket, "price": price,
            "deviation": 10, "magic": position.magic, "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        self.logger.info(f"Sending close order request for position {position_ticket}: {request}")
        order_result = mt5.order_send(request)

        if order_result is None: self.logger.error(f"close_position order_send failed for {position_ticket}. MT5 error: {mt5.last_error()}"); return None
        
        self.logger.info(f"Close order result for pos {position_ticket} ({symbol}): {order_result.comment} (Retcode: {order_result.retcode})")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE: self.logger.info(f"Position {position_ticket} closed/reduced successfully.")
        return order_result

    def modify_position_sltp(self, position_ticket: int, new_sl: typing.Optional[float] = None,
                             new_tp: typing.Optional[float] = None) -> typing.Optional[mt5.TradeResult]:
        """Modifies SL and/or TP for an open position."""
        if not self.is_connected and not self.connect(): return None

        positions = mt5.positions_get(ticket=position_ticket)
        if not positions: self.logger.error(f"Position {position_ticket} not found for SL/TP modification. Error: {mt5.last_error()}"); return None
        position = positions[0]

        sl_to_send = float(new_sl) if new_sl is not None else position.sl # Keep current if None
        tp_to_send = float(new_tp) if new_tp is not None else position.tp # Keep current if None
        
        # MT5 expects 0.0 to remove SL/TP. If new_sl/tp is explicitly 0, use it.
        if new_sl == 0.0: sl_to_send = 0.0
        if new_tp == 0.0: tp_to_send = 0.0
        
        if sl_to_send == position.sl and tp_to_send == position.tp:
            self.logger.info(f"No change in SL/TP for position {position_ticket}. Modification skipped.")
            return None # Or a mock success object if upstream expects a TradeResult

        request = {
            "action": mt5.TRADE_ACTION_SLTP, "position": position.ticket,
            "symbol": position.symbol, # Symbol is required for TRADE_ACTION_SLTP
            "sl": sl_to_send, "tp": tp_to_send,
            "magic": position.magic, # Usually keep original magic
        }
        self.logger.info(f"Sending SL/TP modification request for position {position_ticket}: {request}")
        order_result = mt5.order_send(request)

        if order_result is None: self.logger.error(f"modify_position_sltp order_send failed for {position_ticket}. MT5 error: {mt5.last_error()}"); return None
        
        self.logger.info(f"Modify SL/TP result for pos {position_ticket}: {order_result.comment} (Retcode: {order_result.retcode})")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE: self.logger.info(f"SL/TP for position {position_ticket} modified successfully.")
        return order_result

    def get_open_positions(self, symbol: typing.Optional[str] = None, magic_number: typing.Optional[int] = None) -> list[mt5.PositionInfo]:
        """Fetches open positions, optionally filtered by symbol and/or magic number."""
        if not self.is_connected and not self.connect(): return []
        
        try:
            if symbol and magic_number is not None:
                positions = mt5.positions_get(symbol=symbol)
                if positions: positions = [p for p in positions if p.magic == magic_number]
                else: positions = []
            elif symbol:
                positions = mt5.positions_get(symbol=symbol)
            elif magic_number is not None:
                all_positions = mt5.positions_get()
                if all_positions: positions = [p for p in all_positions if p.magic == magic_number]
                else: positions = []
            else:
                positions = mt5.positions_get()
        except Exception as e:
            self.logger.error(f"Error getting positions (Symbol: {symbol}, Magic: {magic_number}): {e}", exc_info=True); return []

        if positions is None: self.logger.error(f"Failed to get open positions. MT5 error: {mt5.last_error()}"); return []
        
        self.logger.debug(f"Found {len(positions)} open position(s) matching criteria.")
        return list(positions)