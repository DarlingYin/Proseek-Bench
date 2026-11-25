import unittest

from provider_policy import infer_provider, load_provider_policy, resolve_provider_route


class ProviderPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_provider_policy()

    def test_direct_glm_models_use_bigmodel(self):
        for model in ("glm-5.3-flash", "glm-5.3-flashx"):
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model, self.policy), "bigmodel")
                route = resolve_provider_route(model=model)
                self.assertEqual(route.provider, "bigmodel")
                self.assertEqual(route.api_key_env, "GLM_API_KEY")
                self.assertEqual(
                    route.api_base,
                    "https://open.bigmodel.cn/api/paas/v4",
                )

    def test_openrouter_glm_exception_remains_explicit(self):
        route = resolve_provider_route(model="z-ai/glm-5.3-flash")
        self.assertEqual(route.provider, "openrouter")

    def test_qianwen_models_do_not_overlap_glm_family(self):
        self.assertEqual(infer_provider("qwen3.8-flash", self.policy), "qianwen")
        self.assertEqual(infer_provider("deepseek-v4.1-flash", self.policy), "qianwen")

    def test_seed_21_pro_uses_volcengine_coding(self):
        route = resolve_provider_route(model="doubao-seed-2.1-pro")
        self.assertEqual(route.provider, "volcengine_coding")
        self.assertEqual(route.api_key_env, "VOLCENGINE_CODING_API_KEY")
        self.assertEqual(
            route.api_base,
            "https://ark.cn-beijing.volces.com/api/coding/v3",
        )


if __name__ == "__main__":
    unittest.main()
