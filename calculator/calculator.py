import math


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return "Error: Division by zero is not allowed"
    return a / b


def power(a, b):
    return a ** b


def square_root(a):
    if a < 0:
        return "Error: Square root of negative number"
    return math.sqrt(a)


def show_menu():
    print("\n==== Python Calculator ====")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Power")
    print("6. Square Root")
    print("7. Exit")


def calculator():
    while True:
        show_menu()
        choice = input("Select an operation (1-7): ")

        if choice == '7':
            print("Exiting calculator. Goodbye!")
            break

        try:
            if choice in ['1', '2', '3', '4', '5']:
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))

                if choice == '1':
                    print("Result:", add(a, b))
                elif choice == '2':
                    print("Result:", subtract(a, b))
                elif choice == '3':
                    print("Result:", multiply(a, b))
                elif choice == '4':
                    print("Result:", divide(a, b))
                elif choice == '5':
                    print("Result:", power(a, b))

            elif choice == '6':
                a = float(input("Enter number: "))
                print("Result:", square_root(a))

            else:
                print("Invalid choice. Please select a valid option.")

        except ValueError:
            print("Error: Please enter valid numeric values.")


if __name__ == "__main__":
    calculator()
