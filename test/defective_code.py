def calculate_area(radius):
    # Incorrect formula for area of a circle, should be pi * r^2
    area = 2 * 3.14 * radius
    return area

def fetch_data_from_file(file_name):
    # File handling with missing exception handling
    file = open(file_name, 'r')
    content = file.read()
    return content

def main():
    # Incorrect variable type used in mathematical operation
    value = "10"  # String instead of integer
    result = value + 5  # TypeError will occur here
    print(result)

    # Logic error: variable isn't initialized before being used
    print(count)  # count is not defined anywhere

    # Calling the method with an invalid file path
    data = fetch_data_from_file('non_existent_file.txt')  # FileNotFoundError

    radius = 5
    print(calculate_area(radius))  # Should print 3.14 * 5^2, but gets incorrect value
