import pandas as pd
import numpy as np
from faker import Faker
from random import choice, randint, random
from datetime import date, timedelta

fake = Faker('fa_IR')  # Persian locale
np.random.seed(42)

# Define categories
CATEGORIES = [
    "پشتیبانی / شکایت مشتری",
    "درخواست فروش / استعلام قیمت",
    "همکاری / پیشنهاد شراکت",
    "هرزنامه / تبلیغات",
    "نامربوط"
]

# Templates for each category
TEMPLATES = {
    "پشتیبانی / شکایت مشتری": [
        ("گزارش مشکل در {product}",
         """با سلام،
        من در تاریخ {date} {product} را خریداری کردم (شماره فاکتور: {invoice}). 
        متأسفانه با مشکل {problem} مواجه شدم. 
        {request}

        با تشکر
        {name}"""),
        ("شکایت درباره خدمات {company}",
         """{company} محترم،
        از خدمات ضعیف شما در زمینه {service} ناراضی هستم. 
        {complaint_detail}
        خواهشمندم هرچه سریعتر این موضوع را پیگیری نمایید.

        با احترام
        {name}""")
    ],
    "درخواست فروش / استعلام قیمت": [
        ("استعلام قیمت برای {product}",
         """سلام،
        لطفا قیمت و شرایط فروش {product} را با مشخصات زیر اعلام نمایید:
        - تعداد: {quantity}
        - مشخصات فنی: {specs}
        - شرایط پرداخت: {payment}

        {signature}"""),
        ("درخواست خرید عمده",
         """با احترام،
        شرکت ما قصد خرید عمده {product} را دارد.

        لطفا پیش فاکتور و زمان تحویل را اعلام نمایید.

        مدیر خرید
        {company}""")
    ],
    "همکاری / پیشنهاد شراکت": [
        ("پیشنهاد همکاری در زمینه {field}",
         """جناب آقای/سرکار خانم {last_name}،
        با توجه به سابقه درخشان شما در حوزه {industry}، 
        پیشنهاد همکاری در زمینه {project} را داریم. 
        جزئیات پیشنهاد:
        {collab_details}

        منتظر پاسخ شما هستیم.

        با احترام
        {name}
        {position}
        {company}"""),
        ("درخواست سرمایه‌گذاری مشترک",
         """موضوع: مشارکت در پروژه {project_name}

        سلام،
        ما در حال برنامه‌ریزی برای {project_description} هستیم
        و پیشنهاد می‌کنیم با سهم {percentage}% مشارکت نمایید.

        مزایای همکاری:
        - {benefit1}
        - {benefit2}

        برای اطلاعات بیشتر تماس بگیرید:
        تلفن: {phone}
        ایمیل: {email}""")
    ],
    "هرزنامه / تبلیغات": [
        ("!!! فرصت ویژه {offer}",
         """<div dir="rtl">
        ⚡⚡ پیشنهاد شگفت‌انگیز ⚡⚡
        فقط تا پایان {date}:
        {product} با {discount}% تخفیف!

        {urgent_call}

        <a href="{fake_url}">برای خرید کلیک کنید</a>
        </div>"""),
        ("برنده شدید! {prize}",
         """سلام {name}،
        شما در قرعه‌کشی {company} برنده {prize} شدید!

        برای دریافت جایزه خود روی لینک زیر کلیک کنید:
        {fake_url}

        ⏰ این پیشنهاد فقط تا {time} معتبر است!""")
    ],
    "نامربوط": [
        ("موضوع تصادفی {random_subject}",
         """سلام،
        این یک ایمیل تصادفی است درباره {random_topic}.
        {random_detail}

        با تشکر
        {name}"""),
        ("پیام غیرمرتبط {random_number}",
         """با احترام،
        این پیام در مورد {random_event} است.
        {random_request}

        مدیر {random_department}
        {company}""")
    ]
}

# Helper functions
def generate_product():
    products = ["گوشی موبایل", "لپ تاپ", "یخچال", "ماشین لباسشویی",
                "نرم افزار CRM", "خدمات ابری", "دستگاه صنعتی"]
    return choice(products)

def generate_problem():
    problems = ["خرابی قطعه اصلی", "عدم عملکرد صحیح", "تاخیر در تحویل",
                "مشکل در گارانتی", "نقص نرم افزاری"]
    return choice(problems)

def generate_fake_url():
    domains = ["click-here-now", "special-offer-today", "prize-claim"]
    return f"http://{choice(domains)}.com/{fake.uuid4()[:8]}"

def generate_random_subject():
    subjects = ["آب و هوا", "ورزش", "فناوری", "سلامتی", "آموزش", "هنر"]
    return choice(subjects)

def generate_random_topic():
    topics = ["آخرین اخبار", "رویدادهای جاری", "توصیه‌های بهداشتی", "نکات آموزشی", "بررسی‌های فناوری"]
    return choice(topics)

def generate_random_detail():
    details = [
        "این موضوع بسیار جالب است و می‌تواند به شما کمک کند.",
        "لطفا برای اطلاعات بیشتر به وب‌سایت ما مراجعه کنید.",
        "ما در حال برگزاری یک رویداد ویژه هستیم."
    ]
    return choice(details)

def generate_random_event():
    events = ["کنفرانس", "کارگاه", "وبینار", "جشنواره", "نمایشگاه"]
    return choice(events)

def generate_random_request():
    requests = [
        "لطفا ثبت‌نام کنید.",
        "برای شرکت در این رویداد با ما تماس بگیرید.",
        "از شما دعوت می‌شود تا در این برنامه شرکت کنید."
    ]
    return choice(requests)

def generate_random_department():
    departments = ["بازاریابی", "تحقیق و توسعه", "فروش", "منابع انسانی"]
    return choice(departments)

def add_noise(text, noise_level=0.1):
    """Add realistic noise to text"""
    chars = list(text)
    for _ in range(int(len(chars) * noise_level)):
        idx = randint(0, len(chars) - 1)
        chars[idx] = choice([" ", ".", "!", "؟", "،", fake.random_letter()])
    return "".join(chars)

def generate_email(category):
    template = choice(TEMPLATES[category])
    placeholders = {
        'product': generate_product(),
        'date': (date.today() - timedelta(days=randint(1, 30))).strftime("%Y-%m-%d"),
        'invoice': fake.bothify(text="INV-#####"),
        'problem': generate_problem(),
        'request': choice(["لطفا راهنمایی کنید.", "درخواست تعویض دارم.", "نیاز به بازگشت هزینه دارم."]),
        'name': fake.name(),
        'company': fake.company(),
        'service': choice(["پشتیبانی فنی", "تحویل کالا", "خدمات پس از فروش"]),
        'complaint_detail': choice([
            "پاسخگویی مناسب دریافت نکردم.",
            "کارشناسان شما اطلاعات کافی نداشتند.",
            "زمان انتظار برای پاسخ بسیار طولانی بود."
        ]),
        'quantity': f"{randint(1, 100)} دستگاه",
        'specs': choice(["نسخه حرفه ای", "رنگ بندی متنوع", "گارانتی 2 ساله"]),
        'payment': choice(["نقدی", "اقساط 6 ماهه", "چک بانکی"]),
        'signature': choice(["با سپاس", "ارادتمند", "با احترام"]),
        'field': choice(["فناوری اطلاعات", "بازاریابی دیجیتال", "تولید صنعتی"]),
        'last_name': fake.last_name(),
        'industry': choice(["IT", "صنعت خودرو", "کشاورزی"]),
        'project': choice(["توسعه پلتفرم آنلاین", "خط تولید جدید", "پروژه تحقیقاتی"]),
        'collab_details': "\n".join([f"- {fake.sentence()}" for _ in range(3)]),
        'position': choice(["مدیر عامل", "رئیس هیئت مدیره", "سرپرست تیم"]),
        'project_name': fake.catch_phrase(),
        'project_description': fake.bs(),
        'percentage': randint(10, 50),
        'benefit1': fake.sentence(),
        'benefit2': fake.sentence(),
        'phone': fake.phone_number(),
        'email': fake.email(),
        'offer': choice(["فروش ویژه", "تخفیف استثنایی", "حراج پایان فصل"]),
        'fake_url': generate_fake_url(),
        'discount': randint(20, 70),
        'urgent_call': choice(["فقط امروز!", "تا موجودی دارد!", "فقط 5 عدد باقی مانده!"]),
        'prize': choice(["یک خودرو", "10 میلیون تومان", "یک سفر تفریحی"]),
        'time': fake.time(pattern="%H:%M"),
        'random_subject': generate_random_subject(),
        'random_topic': generate_random_topic(),
        'random_detail': generate_random_detail(),
        'random_event': generate_random_event(),
        'random_request': generate_random_request(),
        'random_department': generate_random_department(),
        'random_number': str(randint(1, 1000))
    }

    subject = template[0].format(**placeholders)
    body = template[1].format(**placeholders)

    # Add noise to 30% of emails
    if random() < 0.3:
        body = add_noise(body)
        subject = add_noise(subject)

    # Add HTML tags to 20% of emails
    if random() < 0.2:
        body = f"<div dir='rtl'>{body}</div>"

    return {
        'subject': subject,
        'body': body,
        'category': category
    }

# Generate dataset
emails_per_category = 20000
emails = []

for category in CATEGORIES:
    for _ in range(emails_per_category):
        emails.append(generate_email(category))

df = pd.DataFrame(emails)
df['text'] = df['subject'] + '\n' + df['body']
df['label'] = df['category']

# Save to CSV
output_path = 'emails.csv'
df[['text', 'label']].to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Dataset generated with {len(df)} emails and saved to {output_path}")
