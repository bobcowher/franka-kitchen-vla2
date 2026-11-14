SMOLVLM2_CONFIG_500M = {
        "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        "dtype": "bfloat16",
        "emb_dim": 960,
        }

SMOLVLM2_CONFIG_2_2B = {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "revision": "be92141c8d0ceb98596a923c308ac2bbb4a7c466",
        "dtype": "bfloat16",
        "emb_dim": 2048,
        # Its vision encoder takes 384 px, which is no integer reduction of the 896 archive, so the processor resizes from 512.
        "processor_resize": True,
        }
