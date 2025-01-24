
cinnamonbun = 35
old_cinnamonbun = 35 * 0.40


def purchase_cinnamonbun (amount):
 
 print(amount)

 print(f"Price of Cinnamonbun: {cinnamonbun:.2f}")
 print(f"Price of day old Cinnamonbun: {old_cinnamonbun:.2f}")

 total_cost = amount * old_cinnamonbun

 print(f"Total cost: {total_cost:.2f}")


try:
    purchased_amount = float(input("Enter the number of cinnamanbons you want to purchase: "))
    purchase_cinnamonbun(purchased_amount)
except ValueError:
    print("Error: Please enter a valid number.")