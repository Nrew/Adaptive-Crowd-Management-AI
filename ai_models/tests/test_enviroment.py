from ai_models.enviroment.unity_wrapper import UnityEnvWrapper

# TODO: Test enviroment when its all set up.
def test_enviroment_init():
    env = UnityEnvWrapper(file_name="")
    assert env.behavior_name is not None
    env.close()