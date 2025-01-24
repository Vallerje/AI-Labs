def human_to_dog_years(human_years):
    if human_years < 0:
        return "Error: Age cannot be negative."
    
    if human_years <= 2:
        return f"{human_years} human years is equivalent to {human_years * 10.5} dog years."
    
    dog_years = 2 * 10.5 + (human_years - 2) * 4
    return f"{human_years} human years is equivalent to {dog_years} dog years."

try:
    human_years = float(input("Enter the number of human years: "))
    print(human_to_dog_years(human_years))
except ValueError:
    print("Error: Please enter a valid number.")
