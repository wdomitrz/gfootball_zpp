import gym
import numpy as np
from gfootball.env.observation_preprocessing import \
    MINIMAP_NORM_X_MAX, MINIMAP_NORM_Y_MAX, MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MIN

from gfootball.env import observation_preprocessing


class SMMLayer:
    """Base class for the specification of one SMMLayer used in RichSMM.

    The SMMLayer is used to create one layer of minimap (called SMM as in
    gfootball).

    The core job of this class is to fill one frame with data based on current
    row observation from gfootball enviroment. The function which does that is
    generate.

    In many cases the layer will mark several points on the frame
    with constant value which can be done by overriding only get_points function
    which will be then used to mark returned points with 255.

    In other cases the class will need to override generate method and,
    most probably, use the mark_points static method.
    """

    @staticmethod
    def _calculate_position(x, y, frame,
                            x_range=(MINIMAP_NORM_X_MIN, MINIMAP_NORM_X_MAX),
                            y_range=(MINIMAP_NORM_Y_MIN, MINIMAP_NORM_Y_MAX)):
        """Maps coordinates to SMM position.

        This is helper function used by mark_points to determine position of given
        point. It's based on gfootball/env/observation_preprocessing.py::mark_points

        Args:
          frame: 2-d matrix representing one SMM channel ([y, x])
          x_range: pair (min_x_value, max_x_value) representing space of given x
          y_range: pair (min_y_value, max_y_value) representing space of given y
        """
        x_pos = int((x - x_range[0]) / (x_range[1] - x_range[0]) * frame.shape[1])
        y_pos = int((y - y_range[0]) / (y_range[1] - y_range[0]) * frame.shape[0])
        return (max(0, min(frame.shape[1] - 1, x_pos)),
                max(0, min(frame.shape[0] - 1, y_pos)))

    @staticmethod
    def mark_points(frame, points, value=255):
        """Marks coordinates corresponding to given points.

        Maps points to proper SMM positions using _calculate_position and fills
        them with given value.
        It is based on gfootball/env/observation_preprocessing.py::mark_points
        """
        for px, py in points:
            x, y = SMMLayer._calculate_position(px, py, frame)
            frame[y, x] = value

    def get_points(self, observation):
        """Returns the list of points (x,y) which should be marked by the layer."""
        raise NotImplementedError

    def generate(self, observation, frame):
        """Fills one minimap layer (AKA frame) based on the observation.

        This function is called by RichSMM to fill the frame of zeros with
        appreciate values. The default implementation marks all points returned
        by get_points method with 255 value.
        """
        SMMLayer.mark_points(frame, self.get_points(observation))


class RichSMM(gym.ObservationWrapper):
    """Transforms observation to smm with specified layers.

    Args:
        env: Google Football Environment.
        layers: List of definitions for proceeding layers. Each layer should
            be either single SMMLayer object or list of them. In the second case
            the layers will be applied one on another in the order from the list.
        channel_dimensions: size of the desired smm layers

    """

    def __init__(self, env, layers,
                 channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                     observation_preprocessing.SMM_HEIGHT)):
        super().__init__(env)
        self.smm_layers = list(map(lambda x: x if isinstance(x, list) else [x],
                                   layers))
        self._channel_dimensions = channel_dimensions
        shape = (channel_dimensions[1], channel_dimensions[0], len(layers))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        frames = []
        for layers in self.smm_layers:
            frame = np.zeros(self._channel_dimensions)
            for layer in layers:
                frame = layer.generate(observation, frame)
            frames.append(frame)
        return np.array(frames, dtype=np.uint8)


class TeamSMMLayer(SMMLayer):
    """The base class for SMMLayers using team specific values.

    Args:
        team (str): The team which values will be used ('left' or 'right').
    """

    def __init__(self, team):
        super().__init__()
        if team not in ['left', 'right']:
            raise ValueError("Supported values of 'team' are 'left' and 'right")
        self._team = 'team'

    def extract(self, observation, label=None):
        """Returns team specific value with given label from observation."""
        return observation[self._team + "_team" +
                           ("" if label is None else ("_" + label))]

    def get_points(self, observation):
        raise NotImplementedError


class PlayersPositions(TeamSMMLayer):
    """Marks positions of all players in given team."""

    def get_points(self, observation):
        return self.extract(observation)


class BallPosition(SMMLayer):
    """Marks x,y positions of the ball."""

    def get_points(self, observation):
        return [observation["ball"][:2]]


class YellowCards(TeamSMMLayer):
    """Marks position of all players with yellow card in given team."""

    def get_points(self, observation):
        points = []
        card_info = self.extract(observation, "yellow_card")
        positions = self.extract(observation)
        for i, has_card in enumerate(card_info):
            if has_card == 1:
                points.append(positions[i])
        return points


class ActivePlayer(SMMLayer):
    """Marks position of given active (controlled) player."""

    def __init__(self, player_id=0) -> None:
        super().__init__()
        self.player_id = player_id

    def get_points(self, observation):
        if len(observation['active']) <= self.player_id:
            return []
        return observation['active'][self.player_id]
