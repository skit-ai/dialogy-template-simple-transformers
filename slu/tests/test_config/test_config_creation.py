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
    provider_config = json_config_data_provider.set_config()

    assert actual_config == provider_config
    assert provider_config == config._config


@httpretty.activate(allow_net_connect=False, verbose=True)
def test_api_config_json():

    import os
    import json

    from slu.utils.config import Config, OnStartupClientConfigDataProvider
    from slu import constants as const
    
    with open("config/config.json") as handler:
        axis_config = json.load(handler)

    with open("config/oyo.json") as handler:
        oyo_config = json.load(handler)

    body_resp = [axis_config, oyo_config]

    BUILDER_BACKEND_URL = "http://builder.vernacular.ai"
    url = BUILDER_BACKEND_URL + const.CLIENTS_CONFIGS_ROUTE

    httpretty.register_uri(
        httpretty.GET,
        url,
        body=json.dumps(body_resp)
    )

    client_configs = {
        "booking.inform-assist-1": axis_config,
        "booking.inform-assist-2": oyo_config
    }

    startup_config_data_provider = OnStartupClientConfigDataProvider()
    assert client_configs == startup_config_data_provider.set_config()



