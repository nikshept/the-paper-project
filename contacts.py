from urllib.parse import quote

import pandas as pd

# columns we need from the batch list
BATCH_COLUMNS = ["Id", "First Name", "Email", "Age", "Gender"]

# stand-in participants handed out to test contacts in order, repeating if needed (edit freely)
TEST_PARTICIPANTS = [
    {"utc": "test501", "age": 24, "gender": "Man"},
    {"utc": "test502", "age": 27, "gender": "Woman"},
    {"utc": "test503", "age": 31, "gender": "Man"},
    {"utc": "test504", "age": 22, "gender": "Woman"},
    {"utc": "test505", "age": 29, "gender": ""},
]


# reads a csv or excel file (path or Streamlit upload) into a table
def read_table(file):
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        table = pd.read_csv(file)
    else:
        table = pd.read_excel(file)
    return table.dropna(how="all")


# builds one pre-filled survey link, in the same order as the old Excel formula
def make_link(base_link, utc, institute, age, date, gender):
    if not base_link.endswith("?"):
        base_link = base_link + "?"

    link = base_link
    link += "utc1=" + str(utc)
    link += "&institute=" + quote(institute)
    link += "&age=" + str(age)
    link += "&date=" + date
    link += "&gender=" + gender
    link += "&utc2=" + str(utc)
    return link


# real contact list: one row per participant in the batch list
def make_contacts(batch, institute, base_link, date):
    needed = ["Id", "First Name", "Email", "Age", "Gender"]
    for column in needed:
        if column not in batch.columns:
            raise ValueError(f"Batch list is missing the column: {column}")

    contacts = []
    for _, person in batch.iterrows():
        utc = person["Id"]
        age = str(person["Age"]).split()[0]  # "26 years" -> "26"

        if person["Gender"] == "Male":
            gender = "Man"
        elif person["Gender"] == "Female":
            gender = "Woman"
        else:
            gender = ""

        link = make_link(base_link, utc, institute, age, date, gender)
        contacts.append({"email": person["Email"], "name": person["First Name"], "link": link})

    return pd.DataFrame(contacts)


# test contact list: each test email gets a different stand-in participant
def make_test_contacts(test_emails, institute, base_link, date):
    for column in ["email", "name"]:
        if column not in test_emails.columns:
            raise ValueError(f"Test contacts file is missing the column: {column}")

    contacts = []
    for i, (_, person) in enumerate(test_emails.iterrows()):
        fake = TEST_PARTICIPANTS[i % len(TEST_PARTICIPANTS)]
        link = make_link(base_link, fake["utc"], institute, fake["age"], date, fake["gender"])
        contacts.append({"email": person["email"], "name": person["name"], "link": link})

    return pd.DataFrame(contacts)