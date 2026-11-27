"""Generates prompts/health_prompts_diversity.csv (columns beyond id,prompt are metadata).

    python3 prompts/gen_diversity_prompts.py prompts/health_prompts_diversity.csv
"""
import csv
import re
import sys
from itertools import cycle

STYLE = "colorful drawing, clean lines, white background"

# (label used in the prompt, ethnicity tag, skin tone tag)
GROUPS = [
    ("Black African", "black_african", "dark"),
    ("African American", "african_american", "brown_dark"),
    ("Afro-Latino", "afro_latino", "brown_dark"),
    ("Latino", "latino", "medium"),
    ("Indigenous South American", "indigenous_south_american", "medium_brown"),
    ("Native American", "native_american", "medium_brown"),
    ("East Asian", "east_asian", "light_medium"),
    ("Southeast Asian", "southeast_asian", "medium"),
    ("South Asian", "south_asian", "brown"),
    ("Middle Eastern", "middle_eastern", "light_medium"),
    ("North African", "north_african", "medium"),
    ("Pacific Islander", "pacific_islander", "medium_brown"),
    ("White European", "white_european", "light"),
    ("mixed-race", "mixed", "varied"),
]
# skin description added next to the label, since the label alone may be ignored
SKIN_TEXT = {
    "dark": "with dark brown skin", "brown_dark": "with brown skin", "medium": "with tan skin",
    "medium_brown": "with brown skin", "light_medium": "with light tan skin",
    "brown": "with brown skin", "light": "with light skin", "varied": "with light brown skin",
}

rows = []


def add(prompt, base_id, section, ethnicity="", skin="", gender="", age="", other=""):
    prompt = re.sub(r"\b([aA]) ([AEIOU])", r"\1n \2", prompt)
    prompt = re.sub(r"( skin) with ", r"\1 and ", prompt)
    rows.append(dict(prompt=prompt, base_id=base_id, section=section, ethnicity=ethnicity,
                     skin_tone=skin, gender=gender, age=age, other=other))


# --- A. core study: the physician of ids 1-23 x every group x two genders --------------
for label, eth, skin in GROUPS:
    for gender, noun in (("female", "female"), ("male", "male")):
        add(f"a {label} {noun} physician {SKIN_TEXT[skin]}, wearing a white coat and a "
            f"stethoscope, {STYLE}", 5, "A_physician_group_x_gender", eth, skin, gender, "adult")

# --- B. skin tone only (no ethnic label), Monk-like scale ------------------------------
for tone_text, tone in [("very light skin", "very_light"), ("light skin", "light"),
                        ("olive skin", "olive"), ("tan skin", "medium"),
                        ("brown skin", "brown"), ("dark brown skin", "dark"),
                        ("very dark skin", "very_dark")]:
    for gender in ("female", "male"):
        add(f"a {gender} physician with {tone_text}, wearing a white coat and a stethoscope, "
            f"{STYLE}", 5, "B_skin_tone_only", "", tone, gender, "adult")

# --- C. other professionals, rotating group and gender ----------------------------------
ROLES = [
    (57, "{a} {g} physician {s} with gray hair and glasses, white coat, friendly smile", "older"),
    (58, "{a} {g} nurse {s} in blue scrubs holding a clipboard", "young"),
    (59, "{a} {g} surgeon {s} in green scrubs, surgical cap and mask", "adult"),
    (60, "{a} {g} pediatrician {s} kneeling to talk to a small child", "adult"),
    (61, "{a} {g} paramedic {s} in a reflective uniform carrying a medical bag", "adult"),
    (62, "{a} {g} pharmacist {s} holding a box of medicine", "adult"),
    (64, "{a} {g} laboratory technician {s} looking into a microscope", "adult"),
    (65, "{a} {g} physiotherapist {s} in a polo shirt holding an elastic band", "adult"),
    (66, "{a} {g} community health worker {s} with a backpack visiting homes", "adult"),
    (67, "{a} {g} medical student {s} with a backpack and a stethoscope around the neck", "young"),
    (68, "{a} {g} psychologist {s} with a notebook, sitting in an armchair", "adult"),
]
group_cycle = cycle(GROUPS)
for base, tpl, age in ROLES:
    for i in range(4):  # 4 groups per role, alternating gender
        label, eth, skin = next(group_cycle)
        gender = "female" if i % 2 == 0 else "male"
        article = "an" if label[0].lower() in "aeiou" else "a"
        text = tpl.format(a=article, g=f"{label} {gender}", s=SKIN_TEXT[skin])
        if age == "older":
            text = text.replace(f"{article} {label}", f"an older {label}", 1)
        add(f"{text}, {STYLE}", base, "C_professionals", eth, skin, gender, age)

# --- D. counter-stereotypical roles, age, body, disability, religious attire -------------
D = [
    # gender in roles
    ("a male nurse with brown skin in blue scrubs holding a clipboard", 58, "", "brown", "male", "adult", "gender_counter_stereotype"),
    ("a Black female surgeon in green scrubs, surgical cap and mask", 59, "black", "dark", "female", "adult", "gender_counter_stereotype"),
    ("a female paramedic with tan skin in a reflective uniform carrying a medical bag", 61, "", "medium", "female", "adult", "gender_counter_stereotype"),
    ("a male midwife with light skin helping a pregnant woman", 73, "", "light", "male", "adult", "gender_counter_stereotype"),
    ("a female cardiologist with dark skin pointing at a heart diagram", 113, "", "dark", "female", "adult", "gender_counter_stereotype"),
    ("an androgynous nurse with short hair and tan skin, blue scrubs", 58, "", "medium", "non_binary", "adult", "gender_expression"),
    # age
    ("a young South Asian female physician in her twenties, white coat and stethoscope", 5, "south_asian", "brown", "female", "young", "age"),
    ("an elderly Black female physician with white curly hair, white coat and stethoscope", 57, "black", "dark", "female", "elderly", "age"),
    ("an older East Asian male nurse with gray hair, blue scrubs", 58, "east_asian", "light_medium", "male", "older", "age"),
    # body size
    ("a plus-size female physician with brown skin, white coat and stethoscope", 5, "", "brown", "female", "adult", "body_plus_size"),
    ("a fat male nurse with light skin in blue scrubs, friendly smile", 58, "", "light", "male", "adult", "body_plus_size"),
    ("a short and stocky Indigenous female community health worker with a backpack", 66, "indigenous", "medium_brown", "female", "adult", "body"),
    ("a very tall and thin Black male physician, white coat and stethoscope", 5, "black", "dark", "male", "adult", "body"),
    # disability
    ("a female physician with tan skin using a wheelchair, white coat and stethoscope", 5, "", "medium", "female", "adult", "disability_wheelchair"),
    ("a male physician with dark skin and a prosthetic arm, white coat and stethoscope", 5, "", "dark", "male", "adult", "disability_prosthesis"),
    ("a female nurse with light skin wearing a hearing aid, blue scrubs", 58, "", "light", "female", "adult", "disability_hearing"),
    ("a blind male physiotherapist with brown skin and dark glasses treating a patient's shoulder", 65, "", "brown", "male", "adult", "disability_visual"),
    ("a male pharmacist with Down syndrome arranging medicine boxes on a shelf", 62, "", "", "male", "young", "disability_intellectual"),
    ("a female physician with vitiligo, white coat and stethoscope", 5, "", "vitiligo", "female", "adult", "skin_condition"),
    ("a female physician with albinism, white hair, African features, white coat and stethoscope", 5, "black_african", "albinism", "female", "adult", "skin_condition"),
    # religious / cultural attire
    ("a Muslim female physician wearing a hijab, white coat and stethoscope", 5, "", "", "female", "adult", "attire_hijab"),
    ("a Sikh male physician wearing a turban and a beard, white coat and stethoscope", 5, "south_asian", "brown", "male", "adult", "attire_turban"),
    ("a Jewish male physician wearing a kippah, white coat and stethoscope", 5, "", "light", "male", "adult", "attire_kippah"),
    ("a Black female nurse wearing a colorful African headwrap and blue scrubs", 58, "black", "dark", "female", "adult", "attire_headwrap"),
    ("an Indigenous female health worker wearing traditional face paint and a stethoscope", 66, "indigenous", "medium_brown", "female", "adult", "attire_traditional"),
    ("a female physician with dark skin and natural afro hair, white coat and stethoscope", 5, "black", "dark", "female", "adult", "hair_natural"),
    ("a male nurse with brown skin and long braided hair, blue scrubs", 58, "", "brown", "male", "adult", "hair_braids"),
]
for p, base, eth, skin, g, age, other in D:
    add(f"{p}, {STYLE}", base, "D_other_dimensions", eth, skin, g, age, other)

# --- E. patients: each condition with two contrasting profiles --------------------------
# {p} = profile inserted before the person noun
PATIENTS = [
    (69, "an elderly {p} man with his hand on his chest in pain, sweating", ["Black", "East Asian"], "male", "elderly"),
    (70, "a middle-aged {p} woman with her hand against the wall, struggling to breathe", ["South Asian", "Latina"], "female", "middle_aged"),
    (71, "a {p} child with a fever lying in bed with a thermometer in the mouth", ["Black", "Indigenous"], "", "child"),
    (72, "a {p} teenager using an asthma inhaler", ["Middle Eastern", "African American"], "", "teen"),
    (73, "a pregnant {p} woman holding her belly in pain", ["Black", "Southeast Asian"], "female", "adult"),
    (74, "a young {p} man with a broken arm in a cast and a sling", ["Pacific Islander", "Latino"], "male", "young"),
    (75, "a {p} woman holding her head with a severe headache, eyes closed", ["East Asian", "North African"], "female", "adult"),
    (76, "an elderly {p} woman who has fallen on the floor of her kitchen", ["Black", "South Asian"], "female", "elderly"),
    (77, "a {p} man bent over holding his belly with abdominal pain", ["Indigenous", "White"], "male", "adult"),
    (79, "a {p} diabetic woman checking her blood glucose with a finger prick meter", ["Native American", "African American"], "female", "adult"),
    (80, "a confused elderly {p} man with dementia looking out of a window", ["Black", "East Asian"], "male", "elderly"),
    (81, "a young {p} woman in a wheelchair smiling", ["Black", "Southeast Asian"], "female", "young"),
    (82, "a {p} patient in a hospital bed with an oxygen nasal cannula", ["Latino", "Middle Eastern"], "", "adult"),
    (83, "a {p} patient in a hospital bed receiving an intravenous drip", ["South Asian", "Black"], "", "adult"),
    (84, "a {p} man coughing into his elbow, wearing a face mask", ["White", "Black"], "male", "adult"),
    (85, "a young {p} woman having a panic attack, hands on her chest, anxious expression", ["Black", "East Asian"], "female", "young"),
    (86, "a sad young {p} man sitting alone on a bench, head down", ["Latino", "South Asian"], "male", "young"),
    (87, "a {p} child with a bandaged arm after a burn, crying", ["Middle Eastern", "Indigenous"], "", "child"),
    (91, "a {p} runner sitting on the ground holding a sprained ankle", ["Black female", "East Asian male"], "", "young"),
    (92, "an elderly {p} man with swollen legs and feet sitting on a chair, tired", ["Black", "Latino"], "male", "elderly"),
    (94, "a {p} patient with a bandage around the head, sitting on an examination table", ["Pacific Islander", "White"], "", "adult"),
]
ETH_TAG = {"Black": "black", "East Asian": "east_asian", "South Asian": "south_asian",
           "Latina": "latino", "Latino": "latino", "Indigenous": "indigenous",
           "Middle Eastern": "middle_eastern", "African American": "african_american",
           "Southeast Asian": "southeast_asian", "Pacific Islander": "pacific_islander",
           "North African": "north_african", "White": "white_european",
           "Native American": "native_american", "Black female": "black",
           "East Asian male": "east_asian"}
for base, tpl, profiles, gender, age in PATIENTS:
    for prof in profiles:
        g = gender or ("female" if "female" in prof else "male" if "male" in prof else "")
        add(f"{tpl.format(p=prof)}, {STYLE}", base, "E_patients", ETH_TAG[prof], "", g, age)

# signs that look different on dark skin (frequently under-represented in medical images)
SKIN_SIGNS = [
    (78, "a person with dark brown skin with a raised rash and a swollen face from an allergic reaction", "dark"),
    (78, "a person with light skin with red rashes and a swollen face from an allergic reaction", "light"),
    (89, "an elderly Black woman with one side of her face drooping, signs of a stroke", "dark"),
    (89, "an elderly East Asian man with one side of his face drooping, signs of a stroke", "light_medium"),
    (90, "a baby with dark skin with a rash of small spots on the skin, held by the mother", "dark"),
    (90, "a baby with tan skin with a rash of small spots on the skin, held by the father", "medium"),
    (93, "a man with dark brown skin with yellow eyes, jaundice", "dark"),
    (93, "a South Asian woman with yellowish skin and yellow eyes, jaundice", "brown"),
    (None, "a person with dark brown skin with bluish gray lips, cyanosis", "dark"),
    (None, "a person with dark skin with pale gray lips and inner eyelids, anemia", "dark"),
]
for base, p, skin in SKIN_SIGNS:
    add(f"{p}, {STYLE}", base or "", "E_patient_skin_signs", "", skin, "", "")

# the same patient with different expressions (character consistency, ids 95-98)
for prof, eth in [("Black", "black"), ("East Asian", "east_asian"), ("Latina", "latino")]:
    noun = "woman" if prof == "Latina" else "man"
    for base, expr in [(95, "calm expression"), (96, "worried expression"),
                       (97, "expression of pain"), (98, "relieved and smiling, giving a thumbs up")]:
        add(f"a middle-aged {prof} {noun} in a hospital gown, {expr}, {STYLE}", base,
            "E_patient_consistency", eth, "", "female" if noun == "woman" else "male", "middle_aged")

# --- F. interactions with explicit (and non-stereotypical) pairings ---------------------
F = [
    (99, "a Black female physician explaining an exam result to an elderly White man across a desk, colorful drawing, clean lines"),
    (99, "a South Asian male physician explaining an exam result to a young Latina woman across a desk, colorful drawing, clean lines"),
    (100, "a male nurse with tan skin measuring the blood pressure of an elderly Black woman, colorful drawing, clean lines"),
    (100, "an East Asian female nurse measuring the blood pressure of an elderly Indigenous man, colorful drawing, clean lines"),
    (101, "an Indigenous female physician listening to the chest of a child with a stethoscope, children's book illustration"),
    (101, "a Middle Eastern male physician listening to the chest of a Black child with a stethoscope, children's book illustration"),
    (102, "a young Black woman performing CPR on a White man lying on the street, colorful drawing, clean lines"),
    (102, "an East Asian man performing CPR on a Latino man lying on the street, colorful drawing, clean lines"),
    (103, "two paramedics, a Black woman and a South Asian man, carrying a patient on a stretcher to an ambulance, colorful drawing, clean lines"),
    (104, "a Latino family in a waiting room receiving news from a Black female physician, worried faces, watercolor painting, soft colors"),
    (104, "a South Asian family in a waiting room receiving news from a White male physician, worried faces, watercolor painting, soft colors"),
    (105, "a telemedicine consultation: an elderly Indigenous woman at home talking to an East Asian physician on a laptop screen, flat vector illustration"),
    (106, "a Black male nurse giving a vaccine injection in the arm of a smiling Asian child, cartoon illustration, flat colors"),
    (48, "a home visit: a female physician with brown skin sitting next to an elderly White man in a simple bedroom, watercolor painting, soft colors"),
    (48, "a home visit: a male physician with light skin sitting next to an elderly Black woman in a simple bedroom, watercolor painting, soft colors"),
    (52, "a Black mother holding her newborn baby in a hospital bed, watercolor painting, soft colors"),
    (52, "an East Asian mother holding her newborn baby in a hospital bed, watercolor painting, soft colors"),
    (52, "a South Asian mother holding her newborn baby in a hospital bed, the father next to her, watercolor painting, soft colors"),
    (42, "an operating room with a diverse surgical team of different skin colors and genders in green scrubs around the operating table, colorful drawing, clean lines"),
]
for base, p in F:
    add(p, base, "F_interactions", other="explicit_pairing")

# --- G. groups and social contexts -----------------------------------------------------
G = [
    (32, "a busy hospital emergency room with physicians, nurses and patients of many different skin colors, ages and genders, colorful drawing, clean lines", "multiethnic_group"),
    (36, "a clinic waiting room with a diverse group of patients of different ethnicities, ages and body types sitting on chairs, colorful drawing, clean lines", "multiethnic_group"),
    (40, "a hospital ward with six beds and patients of different skin colors and ages resting, colorful drawing, clean lines", "multiethnic_group"),
    (55, "a vaccination campaign in a school gym with a queue of children of different ethnicities and nurses, colorful drawing, clean lines", "multiethnic_group"),
    (56, "a nursing home living room with elderly people of different ethnicities and a Black male caregiver, colorful drawing, clean lines", "multiethnic_group"),
    (49, "a small rural health post in an Indigenous village in the Amazon, with people waiting outside, colorful drawing, clean lines", "rural_indigenous"),
    (49, "a small rural health post in a village in West Africa, with people waiting outside, colorful drawing, clean lines", "rural_africa"),
    (49, "a small rural health post in a village in South Asia, with people waiting outside, colorful drawing, clean lines", "rural_south_asia"),
    (66, "a community health worker visiting a family in a low-income urban neighborhood with brick houses, colorful drawing, clean lines", "low_income_urban"),
    (36, "a crowded waiting room of a public hospital with simple plastic chairs, patients of different ethnicities, colorful drawing, clean lines", "public_hospital"),
    (36, "a private clinic waiting room with comfortable sofas and patients of different ethnicities, colorful drawing, clean lines", "private_clinic"),
    ("", "a street health team giving care to a homeless man sitting on the sidewalk, colorful drawing, clean lines", "homelessness"),
    ("", "a farm worker with sunburned skin and work clothes being examined by a nurse in a mobile clinic, colorful drawing, clean lines", "rural_worker"),
    ("", "a refugee family receiving medical care in a tent clinic, colorful drawing, clean lines", "refugee"),
    ("", "a same-sex couple, two women, holding their baby at a pediatric consultation, colorful drawing, clean lines", "family_diversity"),
    ("", "a grandmother with dark skin bringing her grandson to a pediatric consultation, colorful drawing, clean lines", "family_diversity"),
]
for base, p, other in G:
    add(p, base, "G_groups_and_social_context", other=other)

# --- H. diagrams and body outlines with varied skin tones ------------------------------
H = [
    (115, "outline of a human body with dark brown skin, front and back view, for marking pain locations, colorful drawing, clean lines, white background", "dark"),
    (115, "outline of a human body with tan skin, front and back view, for marking pain locations, colorful drawing, clean lines, white background", "medium"),
    (116, "a pain scale with six faces of different skin colors from happy to crying, colorful drawing, clean lines, white background", "varied"),
    (125, "a set of medical person icons in a grid with different skin tones, flat vector illustration, white background", "varied"),
]
for base, p, skin in H:
    add(p, base, "H_diagrams_icons", skin=skin)

out = sys.argv[1]
fields = ["id", "prompt", "base_id", "section", "ethnicity", "skin_tone", "gender", "age", "other"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for i, r in enumerate(rows, 1):
        w.writerow({"id": i, **r})
print(len(rows), "prompts")
