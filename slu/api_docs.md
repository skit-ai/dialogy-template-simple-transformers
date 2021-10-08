[Private Postman Collection](https://identity.getpostman.com/login?continue=https%3A%2F%2Fskit-ai.postman.co%2Fworkspace%2FSkit-workspace~9666046d-3b7a-4b78-8f00-c1c19007adfb%2Fdocumentation%2F5007996-b5d9b979-9944-4fb4-bfc3-9ce5039bdbb4)

## Pre Requisites

To run the examples locally, you would need:

1. [kubectx](https://github.com/ahmetb/kubectx/releases)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. AWS credentials
4. Access to [Production](https://outline.skit.ai/doc/how-to-access-the-on-cloud-production-cluster-zfkZKwieRY) or [Staging](https://outline.skit.ai/doc/how-to-access-the-on-cloud-staging-cluster-uvey9GSrhB) cluster.

## Running examples

Spin up your terminal and run:

```shell
kubectl port-forward service/${PROJECT_NAME} 8001:80
```

Refer to the examples to get the value for `$PROJECT_NAME`, it should be one of:

| PROJECT_NAME  | CLIENT_NAME |
|---------------|-------------|
| "vodafone"    | "vodafone"  |
| "dialogy-oyo" | "oyo"       |
| "jarvis"      | "jarvis"    |

Configure the value of `CLIENT_NAME` and `LANGUAGE` in the postman environment.

---
## Request

The structure of the API request is covered within a json-schema snippet below, we will discuss the conceptual requirements to execute the API.

|              | Required | Type Definition                           | Description  |
|--------------|----------|-------------------------------------------|--------------|
| alternatives | Yes      |`List[List[Dict[str, Union[str, float]]]]` | ASR output. This is the required input for a voicebot requiring SLU predictions in production. |
| text         | Yes      | `str`                                     | A text input used during preliminary flow design and tests. |
| context      | Optional | `Dict[str, Any]`                          | While `context` is not required for predictions, if present it can change the results. Within this map, there are certain keys of more importance. We will cover them later. |
| intents_info | No       | `List[Dict[str, Any]]`                    | These were required for ordering and filling slots, not used currently. |
| history      | Optional | `Optional[List[Response]]`                | This is a list of `Response`s previously sent by the SLU service for a given call. Starts at `[]` |

This table describes required fields in the request json. We will now cover each input as per relevance. (Skipping intents_info)

### Alternatives

Alternatives are the primary input to the API. This means, if `alternatives` take precedence over `text` if provided simultaneously.

We are representing an audio containing speech in the above mentioned type `List[List[Dict[str, Union[str, float]]]]`. A very popular question that comes up is the need for nesting the transcriptions in two lists. Here's an example that demonstrates a valid case:

```json
[
  [
    {
      "confidence": 0.74446386,
      "transcript": "to skip to the end of this message press pound the members name is Lego not Bullard this patient is covered by"
    },
    {
      "confidence": 0.7524068,
      "transcript": "to skip to the end of this message press pound the members name is whatnot Bullard this patient is covered by"
    }
  ],
  [
    {
      "confidence": 0.7328612,
      "transcript": " a federal employee program family basic option contract the current effective date of coverage is October 11th 2008 the enrolment code is 112 care first is this members primary insurance"
    },
    {
      "confidence": 0.73193204,
      "transcript": " a federal employee program family basic option contract the current effective date of coverage is October 11th 2008 the enrollment code is 112 care first is this members primary insurance"
    },
    {
      "confidence": 0.7283324,
      "transcript": " a federal employee program family basic option contract the current effective date of coverage is October 11th 2008 the enrolment code is 112 Kerr first is this members primary insurance"
    }
  ],
  [
    {
      "confidence": 0.7523858,
      "transcript": " claims must be submitted within 365 days from the date of service no primary care physician is required for this member know referrals are required no pre determinations are required no pre existing conditions apply there are no lifetime maximum"
    },
    {
      "confidence": 0.7524023,
      "transcript": " claims must be submitted within 365 days from the date of service no primary care physician is required for this member know referrals are required no pre determinations are required no preexisting conditions apply there are no lifetime maximum"
    }
  ]
]
```

We represent transcripts along different chunks owing to pauses in the audio. We call a single chunk an *Utterance*.

#### Utterance

An Utterance is the `List[Dict[str, Union[str, float]]]` within `alternatives`.

|            | type    | description                                                            |
|------------|---------|------------------------------------------------------------------------|
| transcript | `str`   | The text hypothesis for a given audio chunk.                           |
| confidence | `float` | Scores that describe the probability of the transcript being accurate. |

_Trivia: `transcripts` are not always sorted by their corresponding confidence scores._

### Context

This is the call context object with a set of keys which may not be consistent. The API may respond to the presence of a these keys:

|                | type        | description |
|----------------|-------------|-------------|
| expected_slots | `List[str]` | A list of slots that are expected. This helps in understanding if a particular entity can be rejected/accepted if its slot is missing/available. |
| bot_response   | `str`       | The voice-bot's prompt. No clear utility, but MLSEs have tried using it for anaphora resolution. |
| current_intent | `str`       | The tracked intent. We rename the predicted intent with `current_intent` if some conditions apply. Example: predicted intent is "Ia" and have predicted an entity "Eb", then the API produces `current_intent` instead. |
| current_state  | `str`       | A `state` name of the node within the conversation flow. |

### Text

A string that can be used for testing. These are not meant for production use, we support it for faster flow iterations (lest they require an ASR to build flows).

---

## Request JSONSchema

Here's the JSONSchema for the request body.

```json
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "alternatives": {
      "type": "array",
      "items": [
        {
          "type": "array",
          "items": [
            {
              "type": "object",
              "properties": {
                "confidence": {
                  "type": "number"
                },
                "transcript": {
                  "type": "string"
                }
              },
              "required": [
                "confidence",
                "transcript"
              ]
            }
          ]
        }
      ]
    },
    "context": {
      "type": "object",
      "properties": {
        "ack_slots": {
          "type": "array",
          "items": {
             "type": "string"
          }
        },
        "asr_provider": {
          "type": "string"
        },
        "bot_response": {
          "type": "string"
        },
        "call_uuid": {
          "type": "string"
        },
        "current_intent": {
          "type": "string"
        },
        "current_state": {
          "type": "string"
        },
        "set_intent": {
          "type": "object",
          "properties": {}
        },
        "uuid": {
          "type": "string"
        },
        "virtual_number": {
          "type": "string"
        }
      },
    },
    "short_utterance": {
      "type": "object"
    },
    "text": {
      "type": "string"
    }
  },
  "required": [
    "alternatives",
    "text"
  ]
}
```

---

## Response

Just like the Request object, we have a json-schema describing the structure of the Response object later, we will describe the conceptual fields, some type definitions are deferred to their dedicated sections:

|          | Required | type                                                | description                                                                         |
|----------|----------|-----------------------------------------------------|-------------------------------------------------------------------------------------|
| response | Yes      | `Dict[str, Union[List[Intent], List[Entity], str]]` | This is the SLU API's response containing Intents, Entities and the project semver. |
| history  | Optional | `Optional[List[Response]]`                          | This is a set of responses that have been sent out by the SLU services.             |

### Response (key)

The Response body contains a `response` key, it contains:

|          | type           | description                                                           |
|----------|----------------|-----------------------------------------------------------------------|
| intents  | `List[Intent]` | A list of intents ranked by confidence score.                         |
| entities | `List[Entity]` | Tokens in an utterance that should be extracted as per business logic |
| version  | `str`          | Semver that describes the dataset and model versions, only a single version describes both the model and dataset. |

#### Intent

|                   | type                  | description |
|-------------------|-----------------------|---------------|
| alternative_index | `Optional[int]`         | The ASR alternative that produced the intent. |
| name              | `str`                   | The name of the intent. |
| parsers           | `List[str]`             | A list of dialogy plugins that were used to get the prediction. This is key if classifier plugin's output was overridden by another. |
| score             | `float`                 | The confidence score of the predicted intent. |
| slots             | `List[Dict[str, Slot]]` | A list containing relationships between a slot's name and its values. |

##### Slot

The type for a `Slot` is `Dict[str, Union[Name, List[SlotType], List[Entity]]]` where `Name` and `SlotType` are of type `str`, describing the:

- Name of the slot.
- Types of entities the slot can fill.
- The values of entities that were filled.

A sample object for reference:

```json
[
  {
    "name": "payment_date",
    "type": [
      "date",
      "datetime",
      "time",
      "interval"
    ],
    "values": [{
        "alternative_index": 0,
        "body": "on 13th of september",
        "entity_type": "date",
        "grain": "day",
        "parsers": [
          "DucklingPlugin"
        ],
        "range": {
          "end": 42,
          "start": 22
        },
        "score": 0.9,
        "type": "date",
        "value": "2021-09-13T00:00:00.000+05:30"
      }]
  }
]
```

The `values` will contain a `List[Entity]`. `Entity` type will be defined in the next section.

#### Entity

An `Entity` type is a `Dict` containing the following fields and their types.

|                   | Type     | Description |
|-------------------|----------|-----------------------------------------------------|
| alternative_index | `Optional[int]`             | The ASR alternative that produced the entity. Is `None` if the entity wasn't produced from transcripts.                                                                                                                                      |
| body              | `str`                       | The portion of the transcript that was used to derive the entity.                                                                                                                                                                            |
| entity_type       | `str`                       | The type of the entity e.g. `date`, `people`, `number`. It can also be                                                                                                                                                                       |
| grain             | `Optional[str]`             | Required for entities that contain a single timestamp value or a time interval. This describes the smallest unit of time described by the token. Say "month" in case the utterance is "august" but "day" if the utterance was "13th august". |
| parsers           | `List[str]`                 | A list of plugins that operated upon the entity. (creation, aggregation, scoring, etc)                                                                                                                                                       |
| range             | `Dict[str, Dict[str, int]]` | Describes the text range in which the                                                                                                                                                                                                        |
| score             | `Optional[float]`           | The confidence score of the entity extracted.                                                                                                                                                                                                |
| type              | `Optional[str]`             | Applicable only to time entities. Describes if the time entity has a `"single"` value or an `"interval"`.                                                                                                                                    |
| value             | `Union[int, str]`           | We represent numbers and people type entities with `int` and everything else as `str`. Datetimes are represented as [ISO format](https://www.w3.org/TR/NOTE-datetime#Examples).                                                              |

We support the following types of entities:

1. Numeric
    1. People
2. Time
    1. Singular date, time or datetime value
    2. An interval of date, time or datetime
    3. Time duration
3. Amount of money (dollars, rupees)
4. Keyword (arbitrary pattern match types)

---

## Response JSONSchema

The following is the json-schema for the Response body.

```json
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "response": {
      "type": "object",
      "properties": {
        "entities": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "alternative_index": {
                "type": "integer"
              },
              "body": {
                "type": "string"
              },
              "entity_type": {
                "type": "string"
              },
              "grain": {
                "type": "string"
              },
              "parsers": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              },
              "range": {
                "type": "object",
                "properties": {
                  "end": {
                    "type": "integer"
                  },
                  "start": {
                    "type": "integer"
                  }
                },
                "required": [
                  "end",
                  "start"
                ]
              },
              "score": {
                "type": [
                  "null",
                  "number"
                ]
              },
              "type": {
                "type": "string"
              },
              "value": {
                "type": [
                  "integer",
                  "string"
                ]
              },
              "dim": {
                "type": "string"
              },
              "latent": {
                "type": "boolean"
              },
              "slot_names": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              },
              "values": {
                "type": "array",
                "items": {
                  "anyOf": [
                    {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "values": {
                            "type": "string"
                          }
                        },
                        "required": [
                          "values"
                        ]
                      }
                    },
                    {
                      "type": "object",
                      "properties": {
                        "type": {
                          "type": "string"
                        },
                        "value": {
                          "type": [
                            "integer",
                            "string"
                          ]
                        }
                      },
                      "required": [
                        "value"
                      ]
                    }
                  ]
                }
              },
              "origin": {
                "type": "string"
              }
            },
            "required": [
              "alternative_index",
              "body",
              "entity_type",
              "parsers",
              "range",
              "score",
              "type",
              "value"
            ]
          }
        },
        "intents": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "alternative_index": {
                "type": "integer"
              },
              "name": {
                "type": "string"
              },
              "parsers": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              },
              "score": {
                "type": "number"
              },
              "slots": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "name": {
                      "type": "string"
                    },
                    "type": {
                      "type": "array",
                      "items": {
                        "type": "string"
                      }
                    },
                    "values": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "alternative_index": {
                            "type": "integer"
                          },
                          "body": {
                            "type": "string"
                          },
                          "entity_type": {
                            "type": "string"
                          },
                          "grain": {
                            "type": "string"
                          },
                          "parsers": {
                            "type": "array",
                            "items": {
                              "type": "string"
                            }
                          },
                          "range": {
                            "type": "object",
                            "properties": {
                              "end": {
                                "type": "integer"
                              },
                              "start": {
                                "type": "integer"
                              }
                            },
                            "required": [
                              "end",
                              "start"
                            ]
                          },
                          "score": {
                            "type": [
                              "null",
                              "number"
                            ]
                          },
                          "type": {
                            "type": "string"
                          },
                          "value": {
                            "type": [
                              "integer",
                              "string"
                            ]
                          },
                          "_meta": {
                            "type": "object"
                          }
                        },
                        "required": [
                          "alternative_index",
                          "body",
                          "entity_type",
                          "parsers",
                          "range",
                          "score",
                          "type",
                          "value"
                        ]
                      }
                    }
                  },
                  "required": [
                    "name",
                    "type",
                    "values"
                  ]
                }
              }
            },
            "required": [
              "alternative_index",
              "name",
              "parsers",
              "score",
              "slots"
            ]
          }
        },
        "version": {
          "type": "string"
        }
      },
      "required": [
        "entities",
        "intents",
        "version"
      ]
    },
    "status": {
      "type": "string"
    }
  },
  "required": [
    "response",
    "status"
  ]
}
```

---
