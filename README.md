# Timo Digital Bank Case Study - Data Analyst
Author: Hua Dai Nam
Published: April 17, 2024

This is an initial report to management based on an exploratory data analysis (EDA) of the contained information.

Dataset description:
- `account_id`: a code consisting of 8 characters, starting with "ID" followed by unique numbers assigned to each customer.
- `date_of_birth`: the date of birth of each customer.
- `txn_ts`: the timestamp of each transaction.
- `txn_amount`: the amount of each transaction.
- `txn_type_code`: a code representing the type of each transaction.

The business believes that the transaction types contained are:
- Internal Transfers (sender & recipient are both SuperBank customers) 
- Interbank Transfer (recipient is not a SuperBank customer)
- Saving Account Withdrawal (money pulled from savings account to the customer’s current account)
- Auto Recurring Savings Account Contribution (money automatically pulled from savings account to the customer’s current account based on e.g. monthly rule)
- Manual Savings Account Contribution (one-time contribution from a current to a savings account)
- Phone Top-Up (airtime purchase)
- Gift Payments (special transaction type – recipient can but does not have to be a SuperBank customer)