from bayansynthtts import auto_diacritize

texts = [
    "أَهْلًا بِكُمْ في بيانسينث. هذا نظام لتوليد الكلام العربي",
    "مرحباً أنا بيانسينث، نظام لتوليد الكلام العربي",
    "بيانسينث نظام ذكي لتوليد الكلام العربي بجودة عالية",
]
for t in texts:
    print(f"INPUT : {t}")
    print(f"OUTPUT: {auto_diacritize(t)}")
    print()
