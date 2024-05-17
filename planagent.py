from waymax import env, config, dynamics, datatypes
from waymax import dataloader
from jax import numpy as jnp
import jax
from waymax.agents import actor_core
from waymax.agents import sim_agent
from typing import Optional
from waymax import visualization
import mediapy
import cv2
import random
# from google.colab import auth
# auth.authenticate_user()
def show_video(imgs, fps=10):
    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("way.mp4", fourcc, fps, (width, height))
    for img in imgs:
        video_writer.write(img)
    video_writer.release()

class ConstantSimAgentActor(sim_agent.SimAgentActor):
  """Sim agent actor that always returns the same output."""

  DELTA_X = 0.5
  DELTA_Y = 0.6
  DELTA_YAW = 0.1

  def update_trajectory(
      self, state: datatypes.SimulatorState
  ) -> datatypes.TrajectoryUpdate:
    """Just add the constant values to the pose."""
    traj = state.current_sim_trajectory
    return datatypes.TrajectoryUpdate(
        x=traj.x + self.DELTA_X,
        y=traj.y + self.DELTA_Y,
        yaw=traj.yaw + self.DELTA_YAW,
        vel_x=traj.vel_x,
        vel_y=traj.vel_y,
        valid=traj.valid,
    )


def constant_velocity_actor() -> actor_core.WaymaxActorCore:
  agent = ConstantSimAgentActor()

  def select_action(
      params: Optional[actor_core.Params],
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: Optional[jax.Array] = None,
  ) -> actor_core.WaymaxActorOutput:
    del params, actor_state, rng
    action = agent.update_trajectory(state).as_action()
    output = actor_core.WaymaxActorOutput(
        action=action,
        actor_state=None,
        is_controlled=~state.object_metadata.is_sdc,
    )
    output.validate()
    return output

  return actor_core.actor_core_factory(
      init=lambda rng, state: None,
      select_action=select_action,
      name='constant_vel',
  )

# Initialization
dynamics_model = dynamics.InvertibleBicycleModel()
env_config = config.EnvironmentConfig()
scenarios = dataloader.simulator_state_generator(config.WOD_1_1_0_TRAINING)
# waymax_env = env.MultiAgentEnvironment(dynamics_model, env_config)
waymax_env =  env.PlanningAgentEnvironment(
        dynamics_model=dynamics.DeltaGlobal(),
        config= env_config,
        sim_agent_actors=[constant_velocity_actor()],
    )
# Rollout
state = waymax_env.reset(next(scenarios))
total_returns = 0

imgs = []
cnt=0
while not state.is_done:
  action_spec = waymax_env.action_spec()
  random_a = [0,0,0]
  action = datatypes.Action(
      data=jnp.array(random_a),
      valid=jnp.ones(action_spec.valid.shape, dtype=jnp.bool_),
  )
  total_returns += waymax_env.reward(state, action)
  print(total_returns)
  state = waymax_env.step(state, action)
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
  cnt=cnt+1
  if cnt==100:
    break

# mediapy.show_video(imgs, fps=10)
show_video(imgs, fps=10)