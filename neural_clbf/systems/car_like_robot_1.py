"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList


class CarLikeRobot1(ControlAffineSystem):
    """
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1
    THETA = 2
    PSI = 3
    # Control indices
    V = 0
    W = 1

    def __init__(
        self,
        nominal_params: Scenario = {},
        dt: float = 0.1,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "L", "b"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "L", "b"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        return valid

    @property
    def n_dims(self) -> int:
        return CarLikeRobot1.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [CarLikeRobot1.THETA]

    @property
    def n_controls(self) -> int:
        return CarLikeRobot1.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[CarLikeRobot1.X] = 1.0
        upper_limit[CarLikeRobot1.Y] = 1.0
        upper_limit[CarLikeRobot1.THETA] = 10000.0
        upper_limit[CarLikeRobot1.PSI] = 10000.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_controls)

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        obs1_min_x, obs1_max_x = (-0.5, -0.3)
        obs1_min_z, obs1_max_z = (-0.5, -0.3)
        obs1_mask_x = torch.logical_or(x[:, 0] <= obs1_min_x, x[:, 0] >= obs1_max_x)
        obs1_mask_z = torch.logical_or(x[:, 1] <= obs1_min_z, x[:, 1] >= obs1_max_z)
        obs1_mask = torch.logical_or(obs1_mask_x, obs1_mask_z)
        safe_mask.logical_and_(obs1_mask)

        obs2_min_x, obs2_max_x = (-0.5, -0.3)
        obs2_min_z, obs2_max_z = (0.3, 0.5)
        obs2_mask_x = torch.logical_or(x[:, 0] <= obs2_min_x, x[:, 0] >= obs2_max_x)
        obs2_mask_z = torch.logical_or(x[:, 1] <= obs2_min_z, x[:, 1] >= obs2_max_z)
        obs2_mask = torch.logical_or(obs2_mask_x, obs2_mask_z)
        safe_mask.logical_and_(obs2_mask)

        obs3_min_x, obs3_max_x = (0.3, 0.5)
        obs3_min_z, obs3_max_z = (-0.5, -0.3)
        obs3_mask_x = torch.logical_or(x[:, 0] <= obs3_min_x, x[:, 0] >= obs3_max_x)
        obs3_mask_z = torch.logical_or(x[:, 1] <= obs3_min_z, x[:, 1] >= obs3_max_z)
        obs3_mask = torch.logical_or(obs3_mask_x, obs3_mask_z)
        safe_mask.logical_and_(obs3_mask)

        obs4_min_x, obs4_max_x = (0.3, 0.5)
        obs4_min_z, obs4_max_z = (0.3, 0.5)
        obs4_mask_x = torch.logical_or(x[:, 0] <= obs4_min_x, x[:, 0] >= obs4_max_x)
        obs4_mask_z = torch.logical_or(x[:, 1] <= obs4_min_z, x[:, 1] >= obs4_max_z)
        obs4_mask = torch.logical_or(obs4_mask_x, obs4_mask_z)
        safe_mask.logical_and_(obs4_mask)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        obs1_min_x, obs1_max_x = (-0.5, -0.3)
        obs1_min_z, obs1_max_z = (-0.5, -0.3)
        obs1_mask_x = torch.logical_and_(x[:, 0] >= obs1_min_x, x[:, 0] <= obs1_max_x)
        obs1_mask_z = torch.logical_and_(x[:, 1] >= obs1_min_z, x[:, 1] <= obs1_max_z)
        obs1_mask = torch.logical_and_(obs1_mask_x, obs1_mask_z)
        unsafe_mask.logical_or_(obs1_mask)

        obs2_min_x, obs2_max_x = (-0.5, -0.3)
        obs2_min_z, obs2_max_z = (0.3, 0.5)
        obs2_mask_x = torch.logical_and_(x[:, 0] >= obs2_min_x, x[:, 0] <= obs2_max_x)
        obs2_mask_z = torch.logical_and_(x[:, 1] >= obs2_min_z, x[:, 1] <= obs2_max_z)
        obs2_mask = torch.logical_and_(obs2_mask_x, obs2_mask_z)
        unsafe_mask.logical_or_(obs2_mask)

        obs3_min_x, obs3_max_x = (0.3, 0.5)
        obs3_min_z, obs3_max_z = (-0.5, -0.3)
        obs3_mask_x = torch.logical_and_(x[:, 0] >= obs3_min_x, x[:, 0] <= obs3_max_x)
        obs3_mask_z = torch.logical_and_(x[:, 1] >= obs3_min_z, x[:, 1] <= obs3_max_z)
        obs3_mask = torch.logical_and_(obs3_mask_x, obs3_mask_z)
        unsafe_mask.logical_or_(obs3_mask)

        obs4_min_x, obs4_max_x = (0.3, 0.5)
        obs4_min_z, obs4_max_z = (0.3, 0.5)
        obs4_mask_x = torch.logical_and_(x[:, 0] >= obs4_min_x, x[:, 0] <= obs4_max_x)
        obs4_mask_z = torch.logical_and_(x[:, 1] >= obs4_min_z, x[:, 1] <= obs4_max_z)
        obs4_mask = torch.logical_and_(obs4_mask_x, obs4_mask_z)
        unsafe_mask.logical_or_(obs4_mask)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        goal_mask = x.norm(dim=-1) <= 0.141

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        l = 0.1
        d = 0.1

        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        theta = x[:, CarLikeRobot1.THETA]
        psi = x[:, CarLikeRobot1.PSI]

        g[:,CarLikeRobot1.X,CarLikeRobot1.V] = torch.cos(theta)
        g[:,CarLikeRobot1.Y,CarLikeRobot1.V] = torch.sin(theta)
        g[:,CarLikeRobot1.THETA,CarLikeRobot1.V] = torch.tan(psi)/l
        g[:,CarLikeRobot1.PSI,CarLikeRobot1.W] = 1.0

        return g
