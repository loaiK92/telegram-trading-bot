# src/bot/main.py

import asyncio
import websockets
import json
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from src.core.config import TELEGRAM_BOT_TOKEN, ALLTICK_API_KEY
from . import handlers

async def websocket_client():
    """Connects to AllTick using the proven method from the original project."""
    # THIS IS THE CORRECT URL FORMAT FROM YOUR ORIGINAL FILE
    uri = f"wss://quote.tradeswitcher.com/quote-b-ws-api?token={ALLTICK_API_KEY}"
    
    # THIS IS THE CORRECT SUBSCRIPTION MESSAGE
    subscribe_message = {
        "cmd_id": 22004,
        "data": {
            "symbol_list": [{"code": "XAUUSD"}]
        }
    }

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("WebSocket connection successful. Subscribing to XAUUSD...")
                await websocket.send(json.dumps(subscribe_message))
                print("Subscription message sent.")

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        # Check for the correct message type
                        if data.get("cmd_id") == 22998 and 'data' in data:
                            quote = data['data']
                            tick = {
                                "timestamp": quote.get('tick_time'),
                                "price": float(quote.get('price'))
                            }
                            # Ensure tick is valid before updating
                            if tick["timestamp"] and tick["price"]:
                                handlers.signal_generator.update_tick_data(tick)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"Error processing message: {message}, error: {e}")
                        continue

        except Exception as e:
            print(f"WebSocket error: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

def run_bot():
    """Initializes and runs the Telegram bot and the WebSocket client."""
    if not TELEGRAM_BOT_TOKEN or not ALLTICK_API_KEY:
        raise ValueError("API keys not found in .env file!")

    print("Starting bot and WebSocket client...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler('help', handlers.help))
    application.add_handler(CommandHandler("scalp", handlers.scalp))
    application.add_handler(CommandHandler("analyze", handlers.analyze))
    application.add_handler(CommandHandler("settings", handlers.settings))
    
    # Register the button handler
    application.add_handler(CallbackQueryHandler(handlers.button_callback))

    async def main():
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        await websocket_client()

    asyncio.run(main())

if __name__ == '__main__':
    run_bot()