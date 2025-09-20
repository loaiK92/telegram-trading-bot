# src/core/translations.py

texts = {
    'en': {
        'welcome': "Welcome! Use /scalp for a real-time signal or /analyze for market analysis.",

        # Settings
        'settings_prompt': "Please select your preferred language:",
        'language_updated': "✅ Language updated to {language}.",

        # Scalp Report
        'scalp_start': "Analyzing live market data... Please wait.",
        'hold_signal': "Current market conditions are unclear. Recommended action: HOLD.",
        'signal_format': "🚨 New Signal: {signal} 🚨\nConfidence: {confidence:.2f}%",
        'error_common': "An error occurred. Please try again later.",
        'error_analysis': "An error occurred during analysis. This could be due to high server load. Please try again in a moment.",

        # Analyze Report
        'analyze_start': "Performing elite market analysis... Please wait.",
        'elite_report_title': "🏆 **XAUUSD Elite Market Outlook** 🏆",
        'current_price': "Current Price",
        'htf_trend': "Overall HTF Trend",
        'poc': "Point of Control (POC)",
        'zones_header': "Here are the 3 highest-probability zones on the current chart:",
        'zone_header': "▶️ Zone #{} ({}){}",
        'zone_refined': " (Refined 🎯)",
        'setup': "Setup",
        'entry_zone': "Entry Zone",
        'stop_loss': "Stop Loss",
        'tp_1': "TP 1",
        'tp_2': "TP 2",
        'setup_buy': "Look for a BUY entry.",
        'setup_sell': "Look for a SELL entry.",
        'disclaimer_elite': "*Disclaimer: This is a market analysis. Always wait for price action confirmation.*",
        'no_zones_found': "No high-confidence, relevant zones identified on the current chart view.",
        'bullish': "Bullish 🐂",
        'bearish': "Bearish 🐻",

        # Help Text
        'help_title': "**Bot Help & Command Guide**",
        'help_intro': "This is an advanced AI assistant for analyzing and trading XAUUSD.",
        'help_analyze_title': "**/analyze**\n`Strategic Market Outlook`",
        'help_analyze_desc': "• **Purpose:** Provides a high-level analysis using the **1-hour (H1) chart** for strategic planning.\n• **Methodology:** Identifies the 3 most recent, high-probability **Supply and Demand zones** by combining Price Action with Volume Profile analysis.\n• **Output:** A detailed text report with entry zones, SL/TP levels, and a professional chart.",
        'help_scalp_title': "**/scalp**\n`AI-Driven Trade Signals`",
        'help_scalp_desc': "• **Purpose:** Generates precise, short-term entry signals using the **15-minute (M15) chart**.\n• **Methodology:** Driven by a custom-trained **Machine Learning model**, confirmed by technical indicators.\n• **Output:** An actionable **BUY** or **SELL** signal with a precise SL/TP calculated based on current market volatility (ATR).",
        'help_disclaimer': "*Disclaimer: Always use proper risk management. This bot is an analytical tool, not financial advice.*",

    },
    'ar': {
        'welcome': "أهلاً بك! استخدم /scalp للحصول على إشارة تداول مباشرة أو /analyze لتحليل السوق.",

        # Settings
        'settings_prompt': "يرجى اختيار لغتك المفضلة:",
        'language_updated': "✅ تم تحديث اللغة إلى {language}.",

        # Scalp Report
        'scalp_start': "جاري تحليل بيانات السوق المباشرة... يرجى الانتظار.",
        'hold_signal': "ظروف السوق الحالية غير واضحة. الإجراء الموصى به: الانتظار.",
        'signal_format': "🚨 إشارة جديدة: {signal} 🚨\nنسبة الثقة: {confidence:.2f}%",
        'error_common': "حدث خطأ. يرجى المحاولة مرة أخرى لاحقاً.",
        'error_analysis': "حدث خطأ أثناء التحليل. قد يكون هذا بسبب الحمل الزائد على الخادم. يرجى المحاولة مرة أخرى بعد لحظات.",

         # Analyze Report
        'analyze_start': "جاري تنفيذ تحليل النخبة للسوق... يرجى الانتظار.",
        'elite_report_title': "🏆 تقرير النخبة لتوقعات XAUUSD 🏆",
        'current_price': "السعر الحالي",
        'htf_trend': "الاتجاه العام (HTF)",
        'poc': "نقطة التحكم (POC)",
        'zones_header': "فيما يلي أعلى ثلاث مناطق احتمالية على الرسم البياني الحالي:",
        'zone_header': "▶️ المنطقة رقم #{} ({}){}",
        'zone_refined': " (منقّحة 🎯)",
        'setup': "الإعداد",
        'entry_zone': "منطقة الدخول",
        'stop_loss': "وقف الخسارة",
        'tp_1': "الهدف 1",
        'tp_2': "الهدف 2",
        'setup_buy': "ابحث عن دخول شراء.",
        'setup_sell': "ابحث عن دخول بيع.",
        'disclaimer_elite': "إخلاء مسؤولية: هذا تحليل سوقي. انتظر دائمًا تأكيد حركة السعر.",
        'no_zones_found': "لم يتم تحديد مناطق موثوقة وذات صلة على الرسم البياني الحالي.",
        'bullish': "صعودي 🐂",
        'bearish': "هبوطي 🐻",

         # Help Text
        'help_title': "**دليل المساعدة والأوامر**",
        'help_intro': "هذا مساعد ذكاء اصطناعي متقدم لتحليل وتداول الذهب (XAUUSD).",
        'help_analyze_title': "**/analyze**\n`تحليل استراتيجي للسوق`",
        'help_analyze_desc': "• **الغرض:** يقدم تحليلًا عالي المستوى للسوق باستخدام **الرسم البياني للساعة (H1)** للتخطيط الاستراتيجي.\n• **المنهجية:** يحدد آخر 3 مناطق **عرض وطلب** ذات احتمالية نجاح عالية، عبر دمج حركة السعر مع تحليل مؤشر \"فوليوم بروفايل\".\n• **النتيجة:** تقرير نصي مفصل مع مناطق الدخول، وقف الخسارة، وأهداف الربح، بالإضافة إلى رسم بياني يوضح هذه المناطق.",
        'help_scalp_title': "**/scalp**\n`إشارات تداول مدعومة بالذكاء الاصطناعي`",
        'help_scalp_desc': "• **الغرض:** يولد إشارات دخول دقيقة وقصيرة المدى باستخدام **الرسم البياني لـ 15 دقيقة (M15)**.\n• **المنهجية:** يعتمد هذا الأمر على **نموذج تعلم آلة (ML)** تم تدريبه خصيصًا، مع تأكيد من المؤشرات الفنية.\n• **النتيجة:** إشارة **شراء** أو **بيع** قابلة للتنفيذ مع وقف خسارة وهدف ربح يتم حسابهما تلقائيًا بناءً على تقلبات السوق الحالية (ATR).",
        'help_disclaimer': "*إخلاء مسؤولية: استخدم دائمًا إدارة مخاطر مناسبة. هذا البوت أداة تحليلية وليس نصيحة مالية.*",
    }
}

def get_text(key: str, lang: str = 'en', **kwargs) -> str:
    """
    Retrieves a text string from the dictionary in the specified language
    and formats it with any provided arguments.
    """
    text = texts.get(lang, texts['en']).get(key, f"Missing text for key: {key}")
    if kwargs:
        return text.format(**kwargs)
    return text