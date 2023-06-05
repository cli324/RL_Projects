import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Helper function to constrain angles within the range (-pi, pi]
def constrain_angle(angle):
    if angle > np.pi:
        diff = angle - np.pi
        new_angle = angle - np.ceil(diff/(2*np.pi)) * 2 * np.pi
        return new_angle
    elif angle <= -np.pi:
        diff = angle + np.pi
        new_angle = angle - np.floor(diff/(2*np.pi)) * 2 * np.pi
        if new_angle == -np.pi:
            new_angle = np.pi
        return new_angle
    else:
        return angle


# Simulator for a cart-pendulum system
# The pendulum is modeled as a slender rod
class CartPendulumSimulator:
    def __init__(self, z0, tf, control_period_ms, pendulum_params):
        # State is defined as np.array([x, xdot, theta, thetadot]), where x is the position of the base of the pendulum
        # Theta is defined relative to the downward direction, with positive theta corresponding to counter-clockwise
        self._z0 = z0
        self._tf = tf  # Length of time of the simulation (seconds)
        self._control_period_s = control_period_ms / 1000  # Time between updates to the control signal

        self._params = pendulum_params
        self._length = pendulum_params["length"]
        self._g = pendulum_params["g"]

        self._t = 0
        self._z = z0
        self._datastore_index = 0

        number_of_timesteps = int(self._tf / self._control_period_s) + 1  # +1 for the initial state
        self.timesteps = np.zeros((number_of_timesteps,))
        self.action_history = np.zeros((number_of_timesteps-1,))
        self.state_history = np.zeros((number_of_timesteps,self._z0.flatten().shape[0]))
        self.state_history[self._datastore_index] = self._z0
        self._datastore_index += 1

        # Animation variables
        self._fps = 60
        self._animation_datastore_index = 0
        self._animation_data = np.zeros((int(tf*self._fps), 2))  # Stores x and theta for drawing the animation
        x0 = self._z0[0]; theta0 = self._z0[2]
        self._animation_data[self._animation_datastore_index,:] = np.array([x0, theta0])
        self._animation_datastore_index += 1

    # Returns the derivative of the state
    # The parameter x is the current state
    # The parameter u is the control signal, which corresponds to the acceleration of the base of the pendulum
    def _rhs(self, _, z, u):
        xdot = z[1]
        xddot = u
        theta = z[2]
        thetadot = z[3]
        thetaddot = (-3 / 2) * (1 / self._length) * (u * np.cos(theta) + self._g * np.sin(theta))

        return np.array([xdot, xddot, thetadot, thetaddot])

    def simulate_episode(self, policy):
        while not self.episode_finished():
            u = policy(self._t, self._params, self._z)
            self.simulate_one_timestep(u)

    # Returns the tuple (t, z) after taking a single timestep in the simulator
    # t is the current time in the simulator, z is the state
    # Raises an exception if taking the timestep would cause the simulator to exceed the time specified by tf
    # Parameter u is the control signal
    def simulate_one_timestep(self, u):
        if self._t + self._control_period_s > self._tf:
            raise Exception("Simulator cannot take a timestep beyond the maximum time specified by tf")

        # Computing timepoints for which the state should be stored for animation purposes
        animation_period = 1/self._fps
        next_animation_timepoint = self._t + (animation_period - (self._t % animation_period))
        if next_animation_timepoint == self._t:
            next_animation_timepoint = self._t + animation_period

        # Only adding in timepoints that will be within the integration time span
        use_last_timepoint_in_animation = False
        integration_tf = self._t + self._control_period_s
        if next_animation_timepoint <= integration_tf:
            last_animation_timepoint = (integration_tf // animation_period) * animation_period
            timepoints_count = np.round((last_animation_timepoint - next_animation_timepoint) / animation_period) + 1
            timepoints = \
                np.linspace(next_animation_timepoint, last_animation_timepoint, int(timepoints_count)).tolist()
            if last_animation_timepoint == integration_tf:
                use_last_timepoint_in_animation = True
            else:
                timepoints.append(integration_tf)
        else:
            timepoints = [integration_tf,]


        sol = solve_ivp(self._rhs, (self._t, integration_tf), self._z, t_eval=timepoints, args=(u,))
        self._t += self._control_period_s
        self._z = sol.y[:,-1].flatten()
        self._z[2] = constrain_angle(self._z[2])

        self.timesteps[self._datastore_index] = self._t
        self.action_history[self._datastore_index-1] = u  # -1 in the indexing is because each action is associated with the previous state
        self.state_history[self._datastore_index] = self._z
        self._datastore_index += 1

        # Storing data for animation
        if use_last_timepoint_in_animation:
            xs = sol.y[0,:].flatten()
            thetas = sol.y[2,:].flatten()
        else:
            xs = sol.y[0,:-1].flatten()
            thetas = sol.y[2,:-1].flatten()
        count = xs.shape[0]
        self._animation_data[self._animation_datastore_index:self._animation_datastore_index + count, 0] = xs
        self._animation_data[self._animation_datastore_index:self._animation_datastore_index + count, 1] = thetas
        self._animation_datastore_index += count

        return self._t, self._z

    def animate_episode(self):
        fig = plt.figure()
        animation = FuncAnimation(fig, self._draw_frame, frames=self._animation_data.shape[0],
                                  interval=(1/self._fps)*1000)
        plt.show()

    def _draw_frame(self, index):
        x = self._animation_data[index,0]
        theta = self._animation_data[index,1]

        plt.clf()

        # Drawing cart
        cart_width = self._length / 4
        cart_height = cart_width / 2
        cart_x_coords = np.array([x-cart_width/2,x+cart_width/2,x+cart_width/2,x-cart_width/2,x-cart_width/2])
        cart_y_coords = np.array([cart_height/2,cart_height/2,-cart_height/2,-cart_height/2,cart_height/2])
        plt.plot(cart_x_coords, cart_y_coords, color=(0, 0, 0))

        # Drawing pendulum
        plt.plot([x, x + self._length*np.sin(theta)], [0, -self._length*np.cos(theta)], color=(0, 0, 1))

        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.title("Cart-Pendulum Simulation")

        buffer_factor = 1.2
        plt.xlim(x-buffer_factor*self._length,x+buffer_factor*self._length)
        plt.ylim(-buffer_factor * self._length, buffer_factor * self._length)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

    def set_z0(self, z0):
        self._z0 = z0

    def episode_finished(self):
        return self._t + self._control_period_s > self._tf

    def reset(self):
        self._t = 0
        self._z = self._z0
        self._datastore_index = 0

        number_of_timesteps = int(self._tf / self._control_period_s) + 1  # +1 for the initial state
        self.timesteps = np.zeros((number_of_timesteps,))
        self.action_history = np.zeros((number_of_timesteps - 1,))
        self.state_history = np.zeros((number_of_timesteps, self._z0.flatten().shape[0]))
        self.state_history[self._datastore_index] = self._z0
        self._datastore_index += 1

        self._animation_datastore_index = 0
        self._animation_data = np.zeros((int(self._tf * self._fps), 2))  # Stores x and theta for drawing the animation
        x0 = self._z0[0]; theta0 = self._z0[2]
        self._animation_data[self._animation_datastore_index, :] = np.array([x0, theta0])
        self._animation_datastore_index += 1
