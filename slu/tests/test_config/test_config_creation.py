import httpretty


def test_config_yaml():

    import yaml
    from slu.utils.config import Config

    with open("config/config.yaml") as handler:
        loaded_config = yaml.load(handler, Loader=yaml.FullLoader)

    config = Config()

    assert config._config == loaded_config


def test_legacy_config_yaml():

    import yaml
    from slu.utils.config import Config

    with open("config/old_config.yaml") as handler:
        loaded_config = yaml.load(handler, Loader=yaml.FullLoader)

    config = Config(config_path="config/old_config.yaml")

    assert config._config == loaded_config


def test_update_config_json():


    import json

    from slu.utils.config import Config, JSONAPIConfigDataProvider
    
    with open("config/config.json") as handler:
        actual_config = json.load(handler)

    json_config_data_provider = JSONAPIConfigDataProvider(actual_config)
    config = Config(config_data_provider=json_config_data_provider)
    provider_config = json_config_data_provider.give_config_data()

    assert actual_config == provider_config
    assert provider_config == config._config


@httpretty.activate(allow_net_connect=False, verbose=True)
def test_api_config_json():

    import os
    import json

    from slu.utils.config import Config, JSONAPIConfigDataProvider
    
    with open("config/config.json") as handler:
        actual_config = json.load(handler)

    httpretty.register_uri(
        httpretty.GET,
        os.getenv("BUILDER_BACKEND_URL"),
        body=json.dumps(actual_config)
    )

    json_config_data_provider = JSONAPIConfigDataProvider()
    config = Config(config_data_provider=json_config_data_provider)
    assert actual_config == config._config



