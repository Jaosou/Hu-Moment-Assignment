#Catesian Coordinates

class CartesianCoordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"CartesianCoordinates(x={self.x}, y={self.y})"
    
    def to_polar(self):
        import math
        r = math.sqrt(self.x**2 + self.y**2)
        theta = math.atan2(self.y, self.x)
        return PolarCoordinates(r, theta)
    
#Polar Coordinates
class PolarCoordinates:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta

    def __repr__(self):
        return f"PolarCoordinates(r={self.r}, theta={self.theta})"
    
    def to_cartesian(self):
        import math
        x = self.r * math.cos(self.theta)
        y = self.r * math.sin(self.theta)
        return CartesianCoordinates(x, y)
    
# Example usage:
if __name__ == "__main__":
    cart = CartesianCoordinates(3, 4)
    print(cart)
    polar = cart.to_polar()
    print(polar)
    cart_converted = polar.to_cartesian()
    print(cart_converted)