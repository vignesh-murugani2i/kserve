import logging
import os

import pytest
from kubernetes import client
from kubernetes.client import V1ResourceRequirements

from kserve import (KServeClient, V1beta1InferenceService,
                    V1beta1InferenceServiceSpec, V1beta1ModelFormat,
                    V1beta1ModelSpec, V1beta1PredictorSpec, V1beta1SKLearnSpec,
                    constants, V1beta1ExplainerSpec, V1beta1AlibiExplainerSpec)


from ..common.utils import KSERVE_TEST_NAMESPACE, explain, predict


@pytest.mark.slow
def test_predictor_response_headers():
    service_name = "isvc-sklearn-v2"

    predictor = V1beta1PredictorSpec(
        min_replicas=1,
        model=V1beta1ModelSpec(
            model_format=V1beta1ModelFormat(
                name="sklearn",
            ),
            runtime="kserve-sklearnserver",
            image="andyi2it/sklearn-headers:latest",
            storage_uri="gs://seldon-models/sklearn/mms/lr_model",
            resources=V1ResourceRequirements(
                requests={"cpu": "50m", "memory": "128Mi"},
                limits={"cpu": "100m", "memory": "512Mi"},
            ),
        ),
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=service_name, namespace=KSERVE_TEST_NAMESPACE
        ),
        spec=V1beta1InferenceServiceSpec(predictor=predictor),
    )

    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config"))
    kserve_client.create(isvc)
    kserve_client.wait_isvc_ready(
        service_name, namespace=KSERVE_TEST_NAMESPACE)

    response_content, response_headers = predict(
        service_name, "./data/iris_input_v2.json", protocol_version="v2", return_response_headers=True)

    assert "my-header" in response_headers
    assert response_headers["my-header"] == "sample"
    assert response_content["outputs"][0]["data"] == [1, 1]

    kserve_client.delete(service_name, KSERVE_TEST_NAMESPACE)


@pytest.mark.slow
def test_explainer_response_headers():
    service_name = 'isvc-explainer-tabular'
    predictor = V1beta1PredictorSpec(
        sklearn=V1beta1SKLearnSpec(
            image="andyi2it/sklearn-headers:latest",
            storage_uri='gs://kfserving-examples/models/sklearn/1.3/income/model',
            resources=V1ResourceRequirements(
                requests={'cpu': '100m', 'memory': '256Mi'},
                limits={'cpu': '250m', 'memory': '512Mi'}
            )
        )
    )
    explainer = V1beta1ExplainerSpec(
        min_replicas=1,
        alibi=V1beta1AlibiExplainerSpec(
            image="andyi2it/alibiexplainer-headers:latest",
            name='kserve-container',
            type='AnchorTabular',
            storage_uri='gs://kfserving-examples/models/sklearn/1.3/income/explainer',
            resources=V1ResourceRequirements(
                requests={'cpu': '100m', 'memory': '256Mi'},
                limits={'cpu': '250m', 'memory': '512Mi'}
            )
        )
    )

    isvc = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=service_name, namespace=KSERVE_TEST_NAMESPACE),
                                   spec=V1beta1InferenceServiceSpec(predictor=predictor, explainer=explainer))

    kserve_client = KServeClient(
        config_file=os.environ.get("KUBECONFIG", "~/.kube/config"))
    kserve_client.create(isvc)
    try:
        kserve_client.wait_isvc_ready(
            service_name, namespace=KSERVE_TEST_NAMESPACE, timeout_seconds=720)
    except RuntimeError as e:
        logging.info(kserve_client.api_instance.get_namespaced_custom_object("serving.knative.dev", "v1",
                                                                             KSERVE_TEST_NAMESPACE, "services",
                                                                             service_name + "-predictor-default"))
        pods = kserve_client.core_api.list_namespaced_pod(KSERVE_TEST_NAMESPACE,
                                                          label_selector='serving.kserve.io/inferenceservice={}'.format(
                                                              service_name))
        for pod in pods.items:
            logging.info(pod)
        raise e

    response_content = predict(service_name, './data/income_input.json')
    assert (response_content["predictions"] == [0])
    precision, response_headers = explain(
        service_name, './data/income_input.json', return_response_headers=True)
    assert "my-header" in response_headers
    assert response_headers["my-header"] == "sample"
    assert (precision["data"]["precision"] > 0.9)
    kserve_client.delete(service_name, KSERVE_TEST_NAMESPACE)
