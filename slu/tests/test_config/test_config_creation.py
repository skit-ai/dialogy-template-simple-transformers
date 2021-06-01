

def test_config_yaml():

    import yaml
    from slu.utils.config import Config

    with open("config/config.yaml") as handler:
        loaded_config = yaml.load(handler, Loader=yaml.FullLoader)

    config = Config()

    assert config.give_ == loaded_config



def test_config_json():


    from slu.utils.config import Config, JSONAPIConfigDataProvider

    json_config_data_provider = JSONAPIConfigDataProvider()

    config = Config(config_data_provider=json_config_data_provider)
    # assert config._config == 


