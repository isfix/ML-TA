# execution/mt5_interface.py
"""
Interface for interacting with the MetaTrader 5 (MT5) trading platform.
Includes functionalities for connecting, fetching data, and executing trades.
Based on Project 1's MT5Interface.
"""
import MetaTrader5 as mt5
import pandas as pd
import time 
import typing 

try:
    import config
    from utilities import logging_utils # Corrected: Direct import for flat structure
except ImportError:
    print("Warning: Could not perform standard config/utilities import in MT5Interface. Ensure paths are correct.")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    class MockConfig:
        LOG_FILE_LIVE = "mt5_interface_temp.log"; MT5_LOGIN = "demologin"; MT5_PASSWORD = "demopassword"; MT5_SERVER = "demoserver"; MT5_PATH = ""
    if 'config' not in locals() and 'config' not in globals(): config = MockConfig()
    try: import logging_utils as temp_logging_utils; logger = temp_logging_utils.setup_logger(__name__, config.LOG_FILE_LIVE) # type: ignore
    except ImportError: logger = logging.getLogger(__name__)
else: 
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_LIVE)


class MT5Interface:
    def __init__(self, config_obj, logger_obj): 
        self.config = config_obj
        self.logger = logger_obj
        self.is_connected = False
        self.connect()

    def connect(self) -> bool:
        if self.is_connected and mt5.terminal_info() is not None: self.logger.info("Already connected to MT5."); return True
        self.logger.info(f"Attempting to connect to MT5: Login={self.config.MT5_LOGIN}, Server={self.config.MT5_SERVER}")
        mt5_path_to_use = self.config.MT5_PATH if self.config.MT5_PATH else None
        if not mt5.initialize(login=self.config.MT5_LOGIN, password=self.config.MT5_PASSWORD, server=self.config.MT5_SERVER, path=mt5_path_to_use):
            self.logger.error(f"MT5 initialize failed, error code: {mt5.last_error()}"); self.is_connected = False; return False
        self.logger.info(f"MT5 initialized successfully. Version: {mt5.version()}")
        terminal_info = mt5.terminal_info(); 
        if terminal_info: self.logger.info(f"Terminal: {terminal_info.name}, Build: {terminal_info.build}")
        account_info = mt5.account_info()
        if account_info:
            self.logger.info(f"Logged in to MT5 account: {account_info.login}, Server: {account_info.server}, Balance: {account_info.balance:.2f} {account_info.currency}")
            self.is_connected = True; return True
        else:
            self.logger.error(f"Failed to get account info. Error: {mt5.last_error()}."); mt5.shutdown(); self.is_connected = False; return False

    def disconnect(self):
        if mt5.terminal_info(): self.logger.info("Shutting down MT5 connection."); mt5.shutdown()
        else: self.logger.info("MT5 connection already shut down or not initialized.")
        self.is_connected = False

    def get_account_info(self) -> typing.Optional[mt5.AccountInfo]:
        if not self.is_connected and not self.connect(): return None
        info = mt5.account_info()
        if not info: self.logger.error(f"Failed to get account info: {mt5.last_error()}"); return None
        return info

    def get_symbol_info(self, symbol: str) -> typing.Optional[mt5.SymbolInfo]:
        if not self.is_connected and not self.connect(): return None
        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.warning(f"Symbol {symbol} not found. Attempting select.")
            if not mt5.symbol_select(symbol, True): self.logger.error(f"Failed to select symbol {symbol}. Error: {mt5.last_error()}"); return None
            time.sleep(0.2); info = mt5.symbol_info(symbol)
            if info is None: self.logger.error(f"Symbol {symbol} still not found. Error: {mt5.last_error()}"); return None
        return info

    def get_symbol_tick_info(self, symbol: str) -> typing.Optional[mt5.Tick]: 
        if not self.is_connected and not self.connect(): return None
        if not self.get_symbol_info(symbol): self.logger.error(f"Cannot get tick info for {symbol}."); return None
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None: self.logger.error(f"Failed to get tick info for {symbol}: {mt5.last_error()}"); return None
        return tick_info

    def place_market_order(self, symbol: str, order_type_mt5: int, volume: float, sl_price: typing.Optional[float], tp_price: typing.Optional[float], magic_number: int = 0, comment: str = "ai_trade_bot", deviation: int = 10) -> typing.Optional[typing.Any]: 
        if not self.is_connected and not self.connect(): return None
        tick = self.get_symbol_tick_info(symbol)
        if not tick: self.logger.error(f"No tick info for {symbol}."); return None
        price = tick.ask if order_type_mt5 == mt5.ORDER_TYPE_BUY else tick.bid
        if price == 0: self.logger.error(f"Market price for {symbol} is 0. Order cannot be placed."); return None
        sl_to_send = float(sl_price) if sl_price is not None and sl_price > 0 else 0.0
        tp_to_send = float(tp_price) if tp_price is not None and tp_price > 0 else 0.0
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume), "type": order_type_mt5, "price": price, "sl": sl_to_send, "tp": tp_to_send, "deviation": deviation, "magic": magic_number, "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        self.logger.info(f"Sending market order request: {request}")
        order_result = mt5.order_send(request)
        if order_result is None: self.logger.error(f"order_send failed for {symbol}. MT5 error: {mt5.last_error()}"); return None
        self.logger.info(f"Order send result for {symbol}: Code={order_result.retcode}, Comment='{order_result.comment}'")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE: self.logger.info(f"Order executed: Deal ID {order_result.deal} for {symbol}")
        else: self.logger.error(f"Order execution failed for {symbol}: {order_result.comment}"); self.logger.debug(f"Failed request: {request}")
        return order_result

    def close_position(self, position_ticket: int, volume_to_close: typing.Optional[float] = None, comment: str = "ai_bot_close") -> typing.Optional[typing.Any]: 
        if not self.is_connected and not self.connect(): return None
        positions = mt5.positions_get(ticket=position_ticket)
        if not positions: self.logger.error(f"Position {position_ticket} not found. Error: {mt5.last_error()}"); return None
        position = positions[0]; symbol = position.symbol
        close_volume = float(volume_to_close) if volume_to_close is not None and volume_to_close > 0 else position.volume
        if close_volume <= 0 or close_volume > position.volume: self.logger.error(f"Invalid volume {close_volume:.2f} to close for pos {position_ticket}"); return None
        tick = self.get_symbol_tick_info(symbol)
        if not tick: self.logger.error(f"No tick info for {symbol} to close pos {position_ticket}."); return None
        close_order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
        if price == 0: self.logger.error(f"Market price for closing {symbol} is 0."); return None
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": close_volume, "type": close_order_type, "position": position.ticket, "price": price, "deviation": 10, "magic": position.magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        self.logger.info(f"Sending close order request for pos {position_ticket}: {request}")
        order_result = mt5.order_send(request)
        if order_result is None: self.logger.error(f"close_position order_send failed for {position_ticket}. MT5 error: {mt5.last_error()}"); return None
        self.logger.info(f"Close order result for pos {position_ticket}: {order_result.comment}")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE: self.logger.info(f"Position {position_ticket} closed/reduced.")
        return order_result

    def modify_position_sltp(self, position_ticket: int, new_sl: typing.Optional[float] = None, new_tp: typing.Optional[float] = None) -> typing.Optional[typing.Any]:
        if not self.is_connected and not self.connect(): return None
        positions = mt5.positions_get(ticket=position_ticket)
        if not positions: self.logger.error(f"Position {position_ticket} not found. Error: {mt5.last_error()}"); return None
        position = positions[0]
        sl_to_send = float(new_sl) if new_sl is not None else position.sl 
        tp_to_send = float(new_tp) if new_tp is not None else position.tp 
        if new_sl == 0.0: sl_to_send = 0.0
        if new_tp == 0.0: tp_to_send = 0.0
        if sl_to_send == position.sl and tp_to_send == position.tp: self.logger.info(f"No change in SL/TP for pos {position_ticket}."); return None
        request = {"action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "symbol": position.symbol, "sl": sl_to_send, "tp": tp_to_send, "magic": position.magic}
        self.logger.info(f"Sending SL/TP mod request for pos {position_ticket}: {request}")
        order_result = mt5.order_send(request)
        if order_result is None: self.logger.error(f"modify_position_sltp order_send failed. MT5 error: {mt5.last_error()}"); return None
        self.logger.info(f"Modify SL/TP result for pos {position_ticket}: {order_result.comment}")
        if order_result.retcode == mt5.TRADE_RETCODE_DONE: self.logger.info(f"SL/TP for pos {position_ticket} modified.")
        return order_result

    def get_open_positions(self, symbol: typing.Optional[str] = None, magic_number: typing.Optional[int] = None) -> list: 
        if not self.is_connected and not self.connect(): return []
        positions_tuple: typing.Optional[typing.Tuple[mt5.PositionInfo, ...]] = None
        try:
            if symbol and magic_number is not None:
                positions_tuple = mt5.positions_get(symbol=symbol)
                if positions_tuple: return [p for p in positions_tuple if p.magic == magic_number]
                else: return []
            elif symbol: positions_tuple = mt5.positions_get(symbol=symbol)
            elif magic_number is not None:
                positions_tuple = mt5.positions_get()
                if positions_tuple: return [p for p in positions_tuple if p.magic == magic_number]
                else: return []
            else: positions_tuple = mt5.positions_get()
        except Exception as e: self.logger.error(f"Error getting positions: {e}", exc_info=True); return []
        if positions_tuple is None: self.logger.error(f"Failed to get open positions. MT5 error: {mt5.last_error()}"); return []
        self.logger.debug(f"Found {len(positions_tuple)} open position(s).")
        return list(positions_tuple)