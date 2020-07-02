import pandas as pd

subset_data_url = "https://raw.githubusercontent.com/jitender18/IT_Support_Ticket_Classification_with_AWS_Integration/master/latest_ticket_data.csv"
all_data_url = "https://privdatastorage.blob.core.windows.net/github/support-tickets-classification/datasets/all_tickets.csv"

df_subset_data = pd.read_csv(subset_data_url)
# rename ticket message column to body
df_subset_data.columns = [c.lower() for c in df_subset_data]
df_subset_data = df_subset_data.rename(columns={"description": "body"})
assert "body" in df_subset_data and "category" in df_subset_data
df_subset_data.to_csv("subset_tickets.csv")

df_all_data = pd.read_csv(all_data_url)
df_subset_data.to_csv("all_tickets.csv")
