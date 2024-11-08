from pygame.math import Vector2
from math import tan, radians, degrees, copysign
import random
import numpy as np


class Car:
    def __init__(self, x, y, const_velocity, angle=0.0, length=4, max_steering=1.3, max_acceleration=7.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.const_velocity = const_velocity
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 32.5
        self.brake_deceleration = 20
        self.free_deceleration = 1
        self.angular_velocity = 0.0
        # +- 10 km/hr
        self.variation = 0.0

        self.acceleration = 0.0
        self.steering = 0.0

        self.include_next_lane_mechanic = False
        self.next_lane = None
        self.next_lane_steps = None
        self.next_lane_angles = None

    def reset_car(self, rs_pos_list):
        rand_pos = random.choice(rs_pos_list)
        self.angle = rand_pos[2]
        self.position = (rand_pos[0], rand_pos[1])
        self.velocity.x = 0
        self.velocity.y = 0

    def accelerate(self, dt):
        if self.velocity.x < 0:
            self.acceleration = self.brake_deceleration
        else:
            self.acceleration += 10 * dt

    def brake(self, dt):
        if self.velocity.x > 0:
            self.acceleration = -self.brake_deceleration
        else:
            self.acceleration -= 10 * dt

    def handbrake(self, dt):
        if abs(self.velocity.x) > dt * self.brake_deceleration:
            self.acceleration = -copysign(self.brake_deceleration, self.velocity.x)
        else:
            self.acceleration = -self.velocity.x / dt

    def cruise(self, dt):
        if abs(self.velocity.x) > dt * self.free_deceleration:
            self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
        else:
            if dt != 0:
                self.acceleration = -self.velocity.x / dt

    def steer_right(self, dt):
        self.steering -= 180 * dt

    def steer_left(self, dt):
        self.steering += 180 * dt

    def no_steering(self):
        self.steering = 0

    def update(self, dt):
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        self.velocity += (self.acceleration * dt, 0)
        if self.const_velocity == True:
            self.velocity.x = max(20, min(self.velocity.x, self.max_velocity))
            #print(self.velocity.x)
        else:
            # velocity x may be between 20 and 30
            #print(self.velocity.x+self.variation)
            #print(self.max_velocity)
            self.velocity.x = max(-5+self.variation, min(self.velocity.x+self.variation, self.max_velocity))
            #if self.velocity.x > 10:
            #    self.velocity.x = -5+self.variation
            #print()

        #print( max(-self.max_velocity, min(self.velocity.x, self.max_velocity)))
        # this mechanic helps if you have traffic cars that change lane
        if self.include_next_lane_mechanic is True:
            if self.next_lane is not None:
                if self.next_lane - 0.5 <= self.position.x <= self.next_lane + 0.5:
                    self.angle = -90
                    self.position.x = self.next_lane
                    self.next_lane = None
                    self.next_lane_steps = None
                    self.next_lane_angles = None
                else:
                    if self.next_lane_steps is None:
                        if self.next_lane > self.position.x:
                            self.next_lane_steps = np.arange(self.position.x, self.next_lane, 0.1)
                            nr_of_angles = int(self.next_lane_steps.shape[0] / 2)
                            final_angle = -145
                            angle_rate = (final_angle - -90)/nr_of_angles
                            first_angles = np.arange(-90, final_angle, angle_rate)
                            second_angles = np.flip(first_angles)
                            angles = np.concatenate((first_angles, second_angles))
                        else:
                            self.next_lane_steps = np.arange(self.position.x, self.next_lane, -0.1)
                            nr_of_angles = int(self.next_lane_steps.shape[0] / 2)
                            final_angle = -35
                            angle_rate = abs(-90 - final_angle)/nr_of_angles
                            first_angles = np.arange(-90, final_angle, angle_rate)
                            second_angles = np.flip(first_angles)
                            angles = np.concatenate((first_angles, second_angles))

                        self.next_lane_angles = angles

                    if self.next_lane > self.position.x:
                        self.position.x += 1.0
                    elif self.next_lane < self.position.x:
                        self.position.x -= 1.0
                    angle_index = np.where(self.next_lane_steps == self.position.x)
                    if self.next_lane_angles[angle_index].shape[0] != 0:
                        self.angle = self.next_lane_angles[angle_index]
            else:
                self.no_steering()

        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
            self.angular_velocity = np.abs(angular_velocity)
        else:
            angular_velocity = 0
            self.angular_velocity = np.abs(angular_velocity)

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

        if self.angle < 0:
            self.angle = 360 + self.angle
        if self.angle > 360:
            self.angle = self.angle - 360
