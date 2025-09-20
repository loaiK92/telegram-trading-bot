# src/bot/handlers.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import logging
import time
from src.core.config import ALLTICK_API_KEY
from src.core.database import get_user_language, set_user_language
from src.core.translations import get_text
from src.ml.predict import SignalGenerator
from .analysis import get_market_data, generate_elite_report
from .charting import create_elite_chart
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)
signal_generator = SignalGenerator()

# --- COMMAND HANDLERS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_user_language(update.effective_user.id)
    await update.message.reply_text(get_text('welcome', lang))

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_user_language(update.effective_user.id)
    help_text = (
        f"{get_text('help_title', lang)}\n\n"
        f"{get_text('help_intro', lang)}\n\n---\n\n"
        f"{get_text('help_analyze_title', lang)}\n{get_text('help_analyze_desc', lang)}\n\n---\n\n"
        f"{get_text('help_scalp_title', lang)}\n{get_text('help_scalp_desc', lang)}\n\n---\n\n"
        f"{get_text('help_disclaimer', lang)}"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_user_language(update.effective_user.id)
    await update.message.reply_text(get_text('scalp_start', lang))
    try:
        signal, confidence = signal_generator.get_signal()
        if signal == "HOLD":
            response = get_text('hold_signal', lang)
        else:
            response = get_text('signal_format', lang, signal=signal, confidence=confidence)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error in /scalp command: {e}", exc_info=True)
        await update.message.reply_text(get_text('error_common', lang))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = get_user_language(update.effective_user.id)
    status_message = await update.message.reply_text(get_text('analyze_start', lang))
    
    try:
        df_1h = get_market_data(ALLTICK_API_KEY, '1H', 480)
        df_15m = get_market_data(ALLTICK_API_KEY, '15M', 480)

        if df_1h.empty or len(df_1h) < 61:
            await status_message.edit_text('error_insufficient_data')
            return
        
        report, trade_setups = generate_elite_report(df_1h, df_15m, lang=lang) # Pass lang
        
        if trade_setups:
            chart_buffer = create_elite_chart(df_1h, trade_setups, df_1h['EMA50'])
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=chart_buffer, caption=f"Analysis for XAUUSD"
            )
            await context.bot.send_message(chat_id=update.effective_chat.id, text=report, parse_mode=ParseMode.MARKDOWN)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=report, parse_mode=ParseMode.MARKDOWN)
        await status_message.delete()
    except Exception as e:
        logger.error(f"Error in analyze_command: {e}", exc_info=True)
        await status_message.edit_text('error_report')

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a message with language selection buttons."""
    lang = get_user_language(update.effective_user.id)
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 English", callback_data='set_lang_en'),
            InlineKeyboardButton("🇸🇦 العربية", callback_data='set_lang_ar'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(get_text('settings_prompt', lang), reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses the CallbackQuery and updates the language."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press

    # The data is in the format 'set_lang_xx'
    lang_code = query.data.split('_')[-1]
    user_id = query.from_user.id

    set_user_language(user_id, lang_code)
    
    lang_name = "English" if lang_code == 'en' else "العربية"
    await query.edit_message_text(text=get_text('language_updated', lang_code, language=lang_name))