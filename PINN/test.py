
import numpy as np
import matplotlib.pyplot as plt

class Car:
    def __init__(self, brand, color, year, engine_type, owners):
        self.brand = brand
        self.color = color
        self.year = year
        self.engine_type = engine_type
        self.owners = owners 
previous_owners = ["Alice", "Bob", "Charlie"]

mycar = Car("Toyota", "Red", 2020, "Hybrid", previous_owners)
print(f"Brand: {mycar.brand}")
print(f"Color: {mycar.color}")
print(f"Year: {mycar.year}")
print(f"Engine Type: {mycar.engine_type}")
print(f"Previous Owners: {(mycar.owners)}")