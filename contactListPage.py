import pandas as pd
import streamlit as st

from contacts import read_table, make_contacts, make_test_contacts

st.set_page_config(layout="wide")

TEST_FILE = "assets/test_contacts.csv"  # fixed list of test email addresses in the repo

st.header("Generate Contact List")

b1, b2 = st.columns(2)
base_link = b2.text_input("Base survey link")
batch_count = b1.number_input("Number of batches", min_value=1, step=1)

# collect the inputs for every batch
batches = []
for i in range(batch_count):
    st.subheader(f"Batch {i + 1}")
    c1, c2, c3 = st.columns(3)
    batch_file = c1.file_uploader("Batch list (csv or excel)", type=["csv", "xlsx"], key=f"file{i}")
    institute = c2.text_input("Institute name", key=f"institute{i}")
    survey_date = c3.date_input("Survey date", key=f"date{i}")
    batches.append({"file": batch_file, "institute": institute, "date": survey_date.strftime("%d-%b-%Y")})

all_filled = all(b["file"] and b["institute"] for b in batches)
ready = bool(base_link) and all_filled

# build everything when Generate is clicked, and keep it so the download buttons stay visible
if st.button("Generate", disabled=not ready):
    try:
        all_contacts = []
        all_ids = []
        for b in batches:
            batch = read_table(b["file"])
            all_contacts.append(make_contacts(batch, b["institute"], base_link, b["date"]))
            all_ids.append(batch["Id"])
        contacts = pd.concat(all_contacts, ignore_index=True)

        # QR codes are the participant IDs, one per row, for the Generate Surveys page
        qr_codes = pd.DataFrame({"qr_code": pd.concat(all_ids, ignore_index=True)})

        # the test list uses the first batch's institute and date
        first = batches[0]
        test_contacts = make_test_contacts(read_table(TEST_FILE), first["institute"], base_link, first["date"])
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.session_state.results = {
        "contacts": contacts,
        "test_contacts": test_contacts,
        "qr_codes": qr_codes,
        "batch_count": len(batches),
    }

if not ready:
    st.session_state.pop("results", None)

if "results" in st.session_state:
    results = st.session_state.results
    st.success(f"{len(results['contacts'])} contacts from {results['batch_count']} batch(es) "
               f"and {len(results['test_contacts'])} test contacts ready.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download test contact list", results["test_contacts"].to_csv(index=False),
                       file_name="test_contacts.csv", mime="text/csv", width="stretch")
    d2.download_button("Download contact list", results["contacts"].to_csv(index=False),
                       file_name="contacts.csv", mime="text/csv", width="stretch")
    d3.download_button("Download QR codes", results["qr_codes"].to_csv(index=False),
                       file_name="qr_codes.csv", mime="text/csv", width="stretch")