import pandas as pd
import random

# Define categories from your online pharmacy
categories = [
    "Antibiotics", "Cold & Sick", "Vitamin", "Pain Relief", "Allergy & Sinus",
    "Digestive Health", "Skin Care & Dermatology", "Diabetes Care", "Heart & Blood Pressure", "Women's Health",
    "Men's Health", "Eye & Ear Care", "Baby & Child Care", "First Aid & Wound Care", "Respiratory & Asthma",
    "Supplements & Herbal", "Weight Management", "Sexual Wellness", "Mental Health & Sleep", "Pet Medications"
]

# Generate dummy customer data
num_customers = 8000
data = []

for customer_id in range(1, num_customers + 1):
    purchases = {category: random.randint(0, 20) for category in categories}  # Random purchase counts per category
    preferred_category = max(purchases, key=purchases.get)  # The category with highest purchase count
    
    row = {"CustomerID": customer_id, **purchases, "PreferredCategory": preferred_category}
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("customer_purchase_data.csv", index=False)

print("Dummy customer purchase data generated successfully!")
