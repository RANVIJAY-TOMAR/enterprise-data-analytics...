import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(num_records=10000, filename='sample_service_data.xlsx'):
    """
    Generate realistic sample data for testing the analytics platform
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample data lists
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat']
    zones = ['North', 'South', 'East', 'West', 'Central']
    pincodes = ['400001', '110001', '560001', '500001', '600001', '700001', '411001', '380001', '302001', '395001']
    
    service_types = [
        'AC Repair', 'Plumbing', 'Electrical', 'Carpentry', 'Cleaning', 
        'Appliance Repair', 'Pest Control', 'Painting', 'Carpet Cleaning', 'Gardening'
    ]
    
    service_categories = ['Repair', 'Installation', 'Maintenance', 'Emergency', 'Regular']
    
    technician_names = [
        'Rajesh Kumar', 'Amit Singh', 'Suresh Patel', 'Mohan Sharma', 'Vikram Gupta',
        'Ramesh Verma', 'Anil Yadav', 'Prakash Tiwari', 'Sunil Joshi', 'Dinesh Malhotra',
        'Kishan Reddy', 'Harish Mehta', 'Naresh Kapoor', 'Mahesh Saxena', 'Rakesh Agarwal'
    ]
    
    customer_names = [
        'Priya Sharma', 'Rahul Verma', 'Anjali Patel', 'Vikrant Singh', 'Meera Gupta',
        'Arjun Kumar', 'Zara Khan', 'Aditya Joshi', 'Ishita Reddy', 'Karan Malhotra',
        'Aisha Kapoor', 'Rohan Saxena', 'Neha Agarwal', 'Dev Mehta', 'Tanvi Tiwari'
    ]
    
    # Generate base data
    data = []
    
    # Generate dates for the last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for i in range(num_records):
        # Random date within the last year
        random_days = random.randint(0, 365)
        order_date = start_date + timedelta(days=random_days)
        
        # Service completion date (1-7 days after order)
        completion_days = random.randint(1, 7)
        completion_date = order_date + timedelta(days=completion_days)
        
        # Random delays (20% chance of delay)
        if random.random() < 0.2:
            delay_days = random.randint(1, 5)
            completion_date += timedelta(days=delay_days)
        
        # Generate realistic data
        record = {
            'Order_ID': f'ORD{str(i+1).zfill(6)}',
            'Customer_ID': f'CUST{str(random.randint(1, 1000)).zfill(4)}',
            'Customer_Name': random.choice(customer_names),
            'Customer_Phone': f'+91{random.randint(7000000000, 9999999999)}',
            'Customer_Email': f'customer{random.randint(1, 1000)}@example.com',
            
            'Service_Type': random.choice(service_types),
            'Service_Category': random.choice(service_categories),
            'Service_Description': f'Service for {random.choice(service_types)}',
            
            'Technician_ID': f'TECH{str(random.randint(1, 100)).zfill(3)}',
            'Technician_Name': random.choice(technician_names),
            'Technician_Phone': f'+91{random.randint(7000000000, 9999999999)}',
            
            'City': random.choice(cities),
            'Zone': random.choice(zones),
            'Pincode': random.choice(pincodes),
            'Address': f'Address {random.randint(1, 100)}, {random.choice(cities)}',
            
            'Order_Date': order_date,
            'Scheduled_Date': order_date + timedelta(days=random.randint(1, 3)),
            'Completion_Date': completion_date,
            'Created_At': order_date,
            'Updated_At': completion_date,
            
            'Service_Amount': round(random.uniform(500, 5000), 2),
            'Extra_Charges': round(random.uniform(0, 1000), 2) if random.random() < 0.3 else 0,
            'Discount_Amount': round(random.uniform(0, 500), 2) if random.random() < 0.2 else 0,
            'Total_Amount': 0,  # Will be calculated
            
            'Customer_Rating': random.randint(1, 5),
            'Service_Quality_Score': random.randint(1, 10),
            'Technician_Rating': random.randint(1, 5),
            
            'Order_Status': random.choice(['Completed', 'In Progress', 'Cancelled', 'Scheduled']),
            'Payment_Status': random.choice(['Paid', 'Pending', 'Failed']),
            'Payment_Method': random.choice(['Cash', 'Card', 'UPI', 'Net Banking']),
            
            'Priority_Level': random.choice(['Low', 'Medium', 'High', 'Emergency']),
            'Cancellation_Reason': random.choice(['Customer Request', 'Technician Unavailable', 'Weather', 'None']) if random.random() < 0.1 else 'None',
            
            'Customer_Feedback': random.choice([
                'Excellent service!', 'Good work', 'Satisfactory', 'Could be better', 'Poor service'
            ]),
            
            'Technician_Notes': random.choice([
                'Job completed successfully', 'Minor issues resolved', 'Customer satisfied', 'Follow-up required', ''
            ]),
            
            'Is_Emergency': random.choice([True, False]),
            'Is_Repeated_Customer': random.choice([True, False]),
            'Is_Technician_Changed': random.choice([True, False]) if random.random() < 0.1 else False,
            
            'Estimated_Duration_Hours': random.randint(1, 8),
            'Actual_Duration_Hours': 0,  # Will be calculated
            
            'Customer_Satisfaction_Score': random.randint(1, 10),
            'Service_Efficiency_Score': random.randint(1, 10),
            'Overall_Experience_Score': random.randint(1, 10),
            
            'Referred_By': random.choice(['Website', 'App', 'Phone', 'Referral', 'Walk-in']),
            'Service_Channel': random.choice(['Direct', 'Partner', 'Online', 'Offline']),
            
            'Warranty_Period_Days': random.randint(0, 365),
            'Follow_Up_Required': random.choice([True, False]),
            'Next_Service_Date': completion_date + timedelta(days=random.randint(30, 365)) if random.random() < 0.3 else None,
            
            'Customer_Segment': random.choice(['Premium', 'Regular', 'New', 'VIP']),
            'Service_Complexity': random.choice(['Simple', 'Moderate', 'Complex', 'Very Complex']),
            
            'Technician_Specialization': random.choice(['AC', 'Plumbing', 'Electrical', 'General', 'Specialized']),
            'Technician_Experience_Years': random.randint(1, 15),
            
            'Customer_Lifetime_Value': round(random.uniform(1000, 50000), 2),
            'Customer_Since': order_date - timedelta(days=random.randint(0, 1000)),
            
            'Service_Location_Type': random.choice(['Residential', 'Commercial', 'Industrial', 'Office']),
            'Service_Time_Slot': random.choice(['Morning', 'Afternoon', 'Evening', 'Night']),
            
            'Weather_Condition': random.choice(['Sunny', 'Rainy', 'Cloudy', 'Stormy']),
            'Traffic_Condition': random.choice(['Low', 'Medium', 'High', 'Very High']),
            
            'Equipment_Used': random.choice(['Basic Tools', 'Advanced Tools', 'Specialized Equipment', 'Multiple Tools']),
            'Parts_Replaced': random.choice(['None', 'Minor Parts', 'Major Parts', 'Complete Replacement']),
            
            'Safety_Incident': random.choice([True, False]) if random.random() < 0.05 else False,
            'Quality_Check_Passed': random.choice([True, False]),
            'Customer_Complaint': random.choice([True, False]) if random.random() < 0.1 else False,
            
            'Service_Guarantee_Days': random.randint(0, 90),
            'Insurance_Covered': random.choice([True, False]),
            'Tax_Amount': 0,  # Will be calculated
            
            'Customer_Age_Group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
            'Customer_Gender': random.choice(['Male', 'Female', 'Other']),
            
            'Technician_Availability_Score': random.randint(1, 10),
            'Technician_Communication_Score': random.randint(1, 10),
            'Technician_Professionalism_Score': random.randint(1, 10),
            
            'Service_Completion_Time_Minutes': random.randint(30, 480),
            'Travel_Time_Minutes': random.randint(15, 120),
            'Setup_Time_Minutes': random.randint(5, 30),
            
            'Customer_Education_Level': random.choice(['High School', 'Graduate', 'Post Graduate', 'Professional']),
            'Customer_Income_Level': random.choice(['Low', 'Medium', 'High', 'Very High']),
            
            'Service_Season': random.choice(['Summer', 'Monsoon', 'Winter', 'Spring']),
            'Service_Day_Type': random.choice(['Weekday', 'Weekend', 'Holiday']),
            
            'Technician_Team_Size': random.randint(1, 3),
            'Customer_Family_Size': random.randint(1, 6),
            
            'Service_Urgency_Level': random.choice(['Low', 'Medium', 'High', 'Critical']),
            'Customer_Preference_Time': random.choice(['Morning', 'Afternoon', 'Evening', 'Flexible']),
            
            'Technician_Certification_Level': random.choice(['Basic', 'Intermediate', 'Advanced', 'Expert']),
            'Service_Quality_Assurance': random.choice([True, False]),
            
            'Customer_Referral_Count': random.randint(0, 10),
            'Technician_Referral_Count': random.randint(0, 15),
            
            'Service_Innovation_Score': random.randint(1, 10),
            'Customer_Loyalty_Score': random.randint(1, 10),
            'Technician_Productivity_Score': random.randint(1, 10)
        }
        
        # Calculate derived fields
        record['Total_Amount'] = record['Service_Amount'] + record['Extra_Charges'] - record['Discount_Amount']
        record['Tax_Amount'] = round(record['Total_Amount'] * 0.18, 2)  # 18% GST
        record['Actual_Duration_Hours'] = random.randint(record['Estimated_Duration_Hours'] - 1, record['Estimated_Duration_Hours'] + 2)
        record['Actual_Duration_Hours'] = max(1, record['Actual_Duration_Hours'])  # Ensure minimum 1 hour
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(filename, index=False, engine='openpyxl')
    
    print(f"✅ Sample data generated successfully!")
    print(f"📁 File saved as: {filename}")
    print(f"📊 Total records: {len(df):,}")
    print(f"📋 Total columns: {len(df.columns)}")
    print(f"📅 Date range: {df['Order_Date'].min().strftime('%Y-%m-%d')} to {df['Order_Date'].max().strftime('%Y-%m-%d')}")
    print(f"💰 Revenue range: ₹{df['Total_Amount'].min():,.2f} to ₹{df['Total_Amount'].max():,.2f}")
    print(f"⭐ Average rating: {df['Customer_Rating'].mean():.2f}/5")
    
    return df

def generate_multiple_files():
    """Generate multiple sample files with different sizes"""
    
    # Create sample files directory
    os.makedirs('sample_data', exist_ok=True)
    
    # Generate different sized files
    sizes = [
        (1000, 'small_sample.xlsx'),
        (5000, 'medium_sample.xlsx'),
        (10000, 'large_sample.xlsx'),
        (25000, 'xlarge_sample.xlsx')
    ]
    
    for num_records, filename in sizes:
        filepath = os.path.join('sample_data', filename)
        print(f"\n🔄 Generating {filename} with {num_records:,} records...")
        generate_sample_data(num_records, filepath)
    
    print(f"\n🎉 All sample files generated in 'sample_data' directory!")

if __name__ == "__main__":
    print("🚀 Enterprise Data Analytics Platform - Sample Data Generator")
    print("=" * 60)
    
    # Generate a single large file
    generate_sample_data(10000, 'sample_service_data.xlsx')
    
    # Optionally generate multiple files
    print("\n" + "=" * 60)
    response = input("Would you like to generate multiple sample files? (y/n): ")
    if response.lower() == 'y':
        generate_multiple_files()
    
    print("\n✅ Sample data generation complete!")
    print("📝 You can now upload these files to test the analytics platform.") 