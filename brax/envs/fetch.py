# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains an agent to locomote forward."""

from typing import Tuple

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env


class Fetch(env.Env):
  """Fetch trains a dog to run forward."""

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)
    self.torso_idx = self.sys.body.index['Torso']

  def reset(self, rng: jp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'torsoSpeed': zero,
        'torsoIsUp': zero,
        'torsoHeight': zero
    }
    info = {'rng': rng}
    return env.State(qp, obs, reward, done, metrics, info)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    # big reward for torso moving forward (along x, y axis)
    torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
    torso_delta_x = jp.dot(torso_delta, jp.array([1, 0, 0]))
    torso_speed = 1.25 * torso_delta_x / self.sys.config.dt

    # small reward for torso being up
    up = jp.array([0., 0., 1.])
    torso_up = math.rotate(up, qp.rot[self.torso_idx])
    torso_is_up = .1 * self.sys.config.dt * jp.dot(torso_up, up)

    # small reward for torso height
    torso_height = .1 * self.sys.config.dt * qp.pos[0, 2]

    reward = torso_speed + torso_height + torso_is_up

    state.metrics.update(
        torsoSpeed=torso_speed,
        torsoIsUp=torso_is_up,
        torsoHeight=torso_height)

    return state.replace(qp=qp, obs=obs, reward=reward)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Egocentric observation of the dog's body."""
    torso_fwd = math.rotate(jp.array([1., 0., 0.]), qp.rot[self.torso_idx])
    torso_up = math.rotate(jp.array([0., 0., 1.]), qp.rot[self.torso_idx])

    v_inv_rotate = jp.vmap(math.inv_rotate, include=(True, False))

    pos_local = qp.pos - qp.pos[self.torso_idx]
    pos_local = v_inv_rotate(pos_local, qp.rot[self.torso_idx])
    vel_local = v_inv_rotate(qp.vel, qp.rot[self.torso_idx])

    pos_local = jp.reshape(pos_local, -1)
    vel_local = jp.reshape(vel_local, -1)

    contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
    contacts = jp.where(contact_mag > 0.00001, 1, 0)

    return jp.concatenate([
        torso_fwd, torso_up, pos_local, vel_local, contacts
    ])


_SYSTEM_CONFIG = """
  bodies {
    name: "Torso"
    colliders {
      box {
        halfsize { x: 0.75 y: 0.25 z: 0.125 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Shoulders"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.75 z: 0.125 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Hips"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.75 z: 0.125 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Front Right Upper"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Front Right Lower"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }

  bodies {
    name: "Front Left Upper"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Front Left Lower"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Back Right Upper"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Back Right Lower"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Back Left Upper"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Back Left Lower"
    colliders {
      box {
        halfsize { x: 0.25 y: 0.125 z: 0.5 }
      }
    }
    inertia { x: 1 y: 1 z: 1 }
    mass: 1.0
  }
  bodies {
    name: "Ground"
    colliders { plane {} }
    frozen { all: true }
  }
  joints {
    name: "Torso_Shoulders"
    angle_limit { min: -60 max: 60 }
    parent_offset { x: 1.0 }
    child_offset {}
    parent: "Torso"
    child: "Shoulders"
    angular_damping: 35
  }
  joints {
    name: "Torso_Hips"
    angle_limit { min: -60 max: 60 }
    parent_offset { x: -1.0 }
    child_offset {}
    parent: "Torso"
    child: "Hips"
    angular_damping: 35
  }
  joints {
    name: "Shoulders_Front Right Upper"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: -0.875 }
    child_offset { z: 0.375 }
    parent: "Shoulders"
    child: "Front Right Upper"
    angular_damping: 35
  }
  joints {
    name: "Front Right Upper_Front Right Lower"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: 0.25 z: -0.25 }
    child_offset { z: 0.25 }
    parent: "Front Right Upper"
    child: "Front Right Lower"
    angular_damping: 35
  }
  joints {
    name: "Shoulders_Front Left Upper"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: 0.875 }
    child_offset { z: 0.375 }
    parent: "Shoulders"
    child: "Front Left Upper"
    angular_damping: 35
  }
  joints {
    name: "Front Left Upper_Front Left Lower"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: -0.25 z: -0.25 }
    child_offset { z: 0.25 }
    parent: "Front Left Upper"
    child: "Front Left Lower"
    angular_damping: 35
  }
  joints {
    name: "Hips_Back Right Upper"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: -0.875 }
    child_offset { z: 0.375 }
    parent: "Hips"
    child: "Back Right Upper"
    angular_damping: 35
  }
  joints {
    name: "Back Right Upper_Back Right Lower"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: 0.25 z: -0.25 }
    child_offset { z: 0.25 }
    parent: "Back Right Upper"
    child: "Back Right Lower"
    angular_damping: 35
  }
  joints {
    name: "Hips_Back Left Upper"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: 0.875 }
    child_offset { z: 0.375 }
    parent: "Hips"
    child: "Back Left Upper"
    angular_damping: 35
  }
  joints {
    name: "Back Left Upper_Back Left Lower"
    angle_limit { min: -60 max: 60 }
    rotation { z: 90 }
    parent_offset { y: -0.25 z: -0.25 }
    child_offset { z: 0.25 }
    parent: "Back Left Upper"
    child: "Back Left Lower"
    angular_damping: 35
  }
  actuators {
    name: "Torso_Shoulders"
    torque {}
    joint: "Torso_Shoulders"
    strength: 300.0
  }
  actuators {
    name: "Torso_Hips"
    torque {}
    joint: "Torso_Hips"
    strength: 300.0
  }
  actuators {
    name: "Shoulders_Front Right Upper"
    torque {}
    joint: "Shoulders_Front Right Upper"
    strength: 300.0
  }
  actuators {
    name: "Front Right Upper_Front Right Lower"
    torque {}
    joint: "Front Right Upper_Front Right Lower"
    strength: 300.0
  }
  actuators {
    name: "Shoulders_Front Left Upper"
    torque {}
    joint: "Shoulders_Front Left Upper"
    strength: 300.0
  }
  actuators {
    name: "Front Left Upper_Front Left Lower"
    torque {}
    joint: "Front Left Upper_Front Left Lower"
    strength: 300.0
  }
  actuators {
    name: "Hips_Back Right Upper"
    torque {}
    joint: "Hips_Back Right Upper"
    strength: 300.0
  }
  actuators {
    name: "Back Right Upper_Back Right Lower"
    torque {}
    joint: "Back Right Upper_Back Right Lower"
    strength: 300.0
  }
  actuators {
    name: "Hips_Back Left Upper"
    torque {}
    joint: "Hips_Back Left Upper"
    strength: 300.0
  }
  actuators {
    name: "Back Left Upper_Back Left Lower"
    torque {}
    joint: "Back Left Upper_Back Left Lower"
    strength: 300.0
  }
  friction: 0.77459666924
  gravity { z: -9.8 }
  angular_damping: -0.05
  collide_include {
    first: "Front Right Lower"
    second: "Ground"
  }
  collide_include {
    first: "Front Left Lower"
    second: "Ground"
  }
  collide_include {
    first: "Back Right Lower"
    second: "Ground"
  }
  collide_include {
    first: "Back Left Lower"
    second: "Ground"
  }
  dt: 0.02
  substeps: 4
  """

_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "Torso"
  colliders {
    box {
      halfsize { x: 0.75 y: 0.25 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Shoulders"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.75 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Hips"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.75 z: 0.125 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Right Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Right Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}

bodies {
  name: "Front Left Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Front Left Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Right Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Right Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Left Upper"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Back Left Lower"
  colliders {
    box {
      halfsize { x: 0.25 y: 0.125 z: 0.5 }
    }
  }
  inertia { x: 1 y: 1 z: 1 }
  mass: 1.0
}
bodies {
  name: "Ground"
  colliders { plane {} }
  frozen { all: true }
}
joints {
  name: "Torso_Shoulders"
  angle_limit { min: -60 max: 60 }
  parent_offset { x: 1.0 }
  child_offset {}
  parent: "Torso"
  child: "Shoulders"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Torso_Hips"
  angle_limit { min: -60 max: 60 }
  parent_offset { x: -1.0 }
  child_offset {}
  parent: "Torso"
  child: "Hips"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Shoulders_Front Right Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.875 }
  child_offset { z: 0.375 }
  parent: "Shoulders"
  child: "Front Right Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Front Right Upper_Front Right Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Front Right Upper"
  child: "Front Right Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Shoulders_Front Left Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.875 }
  child_offset { z: 0.375 }
  parent: "Shoulders"
  child: "Front Left Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Front Left Upper_Front Left Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Front Left Upper"
  child: "Front Left Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Hips_Back Right Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.875 }
  child_offset { z: 0.375 }
  parent: "Hips"
  child: "Back Right Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Back Right Upper_Back Right Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Back Right Upper"
  child: "Back Right Lower"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Hips_Back Left Upper"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: 0.875 }
  child_offset { z: 0.375 }
  parent: "Hips"
  child: "Back Left Upper"
  stiffness: 5000.0
  angular_damping: 35
}
joints {
  name: "Back Left Upper_Back Left Lower"
  angle_limit { min: -60 max: 60 }
  rotation { z: 90 }
  parent_offset { y: -0.25 z: -0.25 }
  child_offset { z: 0.25 }
  parent: "Back Left Upper"
  child: "Back Left Lower"
  stiffness: 5000.0
  angular_damping: 35
}
actuators {
  name: "Torso_Shoulders"
  torque {}
  joint: "Torso_Shoulders"
  strength: 300.0
}
actuators {
  name: "Torso_Hips"
  torque {}
  joint: "Torso_Hips"
  strength: 300.0
}
actuators {
  name: "Shoulders_Front Right Upper"
  torque {}
  joint: "Shoulders_Front Right Upper"
  strength: 300.0
}
actuators {
  name: "Front Right Upper_Front Right Lower"
  torque {}
  joint: "Front Right Upper_Front Right Lower"
  strength: 300.0
}
actuators {
  name: "Shoulders_Front Left Upper"
  torque {}
  joint: "Shoulders_Front Left Upper"
  strength: 300.0
}
actuators {
  name: "Front Left Upper_Front Left Lower"
  torque {}
  joint: "Front Left Upper_Front Left Lower"
  strength: 300.0
}
actuators {
  name: "Hips_Back Right Upper"
  torque {}
  joint: "Hips_Back Right Upper"
  strength: 300.0
}
actuators {
  name: "Back Right Upper_Back Right Lower"
  torque {}
  joint: "Back Right Upper_Back Right Lower"
  strength: 300.0
}
actuators {
  name: "Hips_Back Left Upper"
  torque {}
  joint: "Hips_Back Left Upper"
  strength: 300.0
}
actuators {
  name: "Back Left Upper_Back Left Lower"
  torque {}
  joint: "Back Left Upper_Back Left Lower"
  strength: 300.0
}
friction: 0.77459666924
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "Front Right Lower"
  second: "Ground"
}
collide_include {
  first: "Front Left Lower"
  second: "Ground"
}
collide_include {
  first: "Back Right Lower"
  second: "Ground"
}
collide_include {
  first: "Back Left Lower"
  second: "Ground"
}
dt: 0.02
substeps: 4
dynamics_mode: "legacy_euler"
"""
