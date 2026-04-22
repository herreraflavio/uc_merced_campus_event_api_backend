import requests
import time
import json

# --- CONFIGURATION ---
TARGET_URL = "http://10.34.48.10:8080/ai"

# Reusable list of all item IDs to send in the payload for every request
PAYLOAD_ITEM_IDS = ["37e0393e-a2ea-40d3-9a59-99b9c2cf2fa8", "23b4f06a-978e-4ae3-bcdd-ac24478239bb", "edd98591-b305-4892-ad78-be8e7393be73", "64606d38-8948-4294-8211-b074ab6f85b9", "2107794c-d1f0-43f4-8bfd-35cbab79d94b", "a6bebe47-f29c-43c0-b30c-0878e147ec8a", "0558960d-5ef6-4c54-874c-11dfc5d67c99", "c2ab74f7-cb80-4c5a-93de-5875c976f9f8", "dc25b8af-50f3-467a-92c0-dc18bef4536a", "307bf776-b241-4d48-a038-130dfecff379", "4b71acdd-20d9-4483-9042-d35b63815fe7", "1a6664a1-a68e-4111-b91d-9ed41ebbc600", "6bce43e4-0ac9-4fbe-8d13-57f687497a04", "9e5067f9-7724-46b7-9870-4989ebb656e3", "1604efd2-fef8-4102-9d39-36f0dbf80b74", "9bac3479-3ce9-4fe5-8786-e6894a47f1ed", "262fa1fb-519a-49b9-8cfe-0c5546c7d720", "b4aea4fc-5336-4f38-bd3d-cbb743ae53fd", "4060100d-a12c-4469-810b-59152677c0c1", "67afece0-b49a-41cc-aae9-399fd5ab5c1a", "bf3ffa1f-3e29-468f-8a4b-ff9561cabd93", "9abc2e36-6174-491a-9e6e-3c63a36b67ba", "a1c5a6a0-701e-4770-b9ce-698b42430877", "45dea73a-b334-4f72-b7f4-1467650a57a1", "07d7906f-f552-41cb-82f8-e838334dd192", "e1442fe3-ac64-4ebd-ba99-23227060d5da", "e94b8bf5-f672-4a80-a600-7b9eb8efd3a0", "5376b319-9f79-4f9c-91c9-a2fcb6d07e88", "ae204fa2-692f-4d75-b31e-c5e9e0132c3f", "bc0e04ca-b63a-4e7c-a88d-b7381b85d3ed", "4118e2f1-d800-4e50-9a46-8c0ceaf0dfc9", "8ac4261c-49a5-415a-91c7-b048d4d5fff3", "d8b37849-4d71-460f-a7f6-2d1524d62dd5", "b968b7ea-b4c3-4182-a2ae-6d0bef0768ee", "4c61487d-df92-4714-8102-1fef9b489b14", "d0ebe4a9-a66a-4ac4-92e9-3c70699833fa", "b19d40e0-d5a3-412b-b60b-bea9a980fa9b", "5d5754dd-8c59-4ddf-97e0-46bc57622e0f", "941b1e2d-2528-4175-8fc2-a58dc732ee11", "9e9c4574-3eec-40f4-bade-757734bf196e", "c651e5c0-6d64-43d8-ac79-41d17693b0d1", "8a027686-07b3-4d61-8c03-477889b957cf", "b9534dbc-d242-4aae-9310-1dbac1a7e0c6", "3d970001-bb86-424d-9520-7339e8b03cdf", "f1256437-5f5a-4f0a-8e66-530161ee7a62", "e6e3d1b7-ea57-4684-9049-a378aad03f44", "3a0ac5b9-b295-4876-9dd0-ce4f710b185d", "7eaed20f-e778-4053-98aa-faf7c1411d67", "332f984e-f157-40a3-af6d-528897b37a50", "e329d348-ef86-48e5-8403-3753dd567022", "c170f878-faed-40d7-8f05-927fd7ef772c", "62e4c0c0-76d3-4cf2-adae-d81d78709dab", "d21d8acc-d9df-4265-88b9-8340c5ffbf61", "52739c88-3663-4739-9050-d78eedcd0d1a", "994bcf11-6d7c-4909-897e-65c201a0929d", "c2db5c34-e008-4dbe-b87c-235c33ffd4f7", "c974b291-e027-44cf-be24-1b9cdad7a9c9", "89af164a-073b-4b9e-b4cb-f6c35026196a", "74455046-ba7a-4294-a595-17af823874dc", "33714ecd-f57c-4c27-83c5-de0459501bde", "8f31296d-37af-4d9e-bce2-97371d171157", "55b30004-115f-4560-94e4-2b19eea97422", "b709c036-4bee-4f21-ac0c-3bfcb9675fa3", "a836fa04-938f-4b8b-95ca-7970b0cc7ce4", "f868c901-0d41-4acf-be14-8f4d4240c954", "edec728e-f9d9-4f89-80bb-8b268f71d5d8", "807d9a1a-6f2f-461b-a8c8-3e34ce387bf8", "d9be7aa3-a4dd-4929-a37c-fad9d01c51fe",
                    "2aa5039f-b77e-4d67-a8bc-9cb379a55070", "503ecfea-b998-4a82-8577-dfb9e155c851", "983a90c9-341d-4a89-a977-63206a858234", "0169b2d9-547f-4ef5-8621-68058eb43265", "5c1638a5-cb77-4680-b96d-7cf47f4ecb1e", "55ec137e-de3e-4970-a591-455784cff966", "76596545-7f89-4ecd-b2bf-a70995f55423", "9d577385-b0d5-40b0-a3ab-921e3c9ed1c8", "8678dbd1-1bcc-442f-a24e-7d4d0e36b5dc", "5f8062f6-9de2-4b80-81f1-16ed314a85bd", "6944caa9-33cb-4f87-85b3-1a050e48f37d", "df39d0e2-ce6a-4c28-836b-03fc4f169cce", "3b3f27ea-b0a8-4595-b2ed-81d0e9975541", "93717f91-4d83-4a5f-b567-9098f4adc784", "3ec7fcfe-43c1-4ca5-b618-f33fb3078bff", "15ba12d8-fb12-4cd5-8459-00f607f77fc5", "polygon1761858238549970", "polygon1761858397559679", "polygon1761858443392270", "polygon1761858529192827", "polygon1761858559824132", "polygon1761858687866950", "polygon1761858712943575", "polygon1761860170632272", "polygon1761860301580820", "polygon1761860370535493", "polygon1761860435344370", "polygon1761860486750884", "polygon1761860579793939", "polygon176186065863142", "polygon1761860798659497", "polygon1761861346599768", "polygon1761861405251545", "polygon1761861562792597", "polygon1761861620897565", "polygon1761861665664626", "polygon1761861712013544", "polygon1761861787851598", "polygon1761861841336199", "polygon1761861884490287", "polygon1761861947974663", "polygon1761862172959950", "polygon1761862210606822", "polygon1761862248859831", "polygon1761862292593315", "polygon1761862422686169", "polygon1761862844956290", "polygon1761862947648773", "polygon1761863007636326", "polygon1761863114066280", "polygon1761863167448187", "polygon1761863295367247", "polygon1761863506341424", "polygon1761863605342500", "polygon1761861907489943", "polygon1761863691369604", "polygon1761863882489923", "polygon1761864029774549", "polygon17618640488354", "polygon1761864638825198", "polygon176186469807932", "polygon1761864721100301", "polygon1761864744781620", "polygon1761864796496150", "polygon1761860891689236", "polygon1761864842379792", "polygon1761864857677247", "polygon1761864875304656", "polygon1761864911744722", "polygon1761864928221163", "polygon1761864944370381", "polygon1761864963879916", "polygon1761864982860988", "polygon1761864997676827", "polygon1761865011359551", "polygon1761865240288559", "polygon1761865367414523", "polygon1761865382252840", "polygon1761865648911798", "polygon1761866033714580", "polygon1761866231195496", "polygon1761864411289394", "polygon1761941210659436", "polygon1762548133999165", "polygon1762549587833659", "polygon1762549876586356", "polygon176186615563270", "polygon1762551985858856", "polygon1761865825162867", "polygon176255236834235", "polygon1762553384644206", "polygon1761858609026974", "polygon1763160787041855", "polygon1763161880780312"]
# 20 Diverse test cases covering food, events, navigation, and amenities
TEST_CASES = [
    {"query": "where can i get food", "expected_ids": ["id1", "id2"]},
]


def run_benchmarks():
    print(f"🚀 Initiating Batch Benchmark Test to {TARGET_URL}...")
    headers = {"Content-Type": "application/json"}

    total_latency = 0
    total_precision = 0
    total_recall = 0
    successful_runs = 0

    for i, test in enumerate(TEST_CASES, 1):
        query = test["query"]
        expected_ids = test["expected_ids"]

        payload = {
            "query": query,
            "item_ids": PAYLOAD_ITEM_IDS
        }

        print(f"\n[{i}/20] Query: '{query}'")
        start_time = time.perf_counter()

        try:
            response = requests.post(TARGET_URL, json=payload, headers=headers)
            response.raise_for_status()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            data = response.json()
            ranked_ids = data.get("ranked_item_ids", [])

            expected_set = set(expected_ids)
            received_set = set(ranked_ids)
            matches = expected_set.intersection(received_set)

            precision = (len(matches) / len(received_set)
                         * 100) if received_set else 0.0
            recall = (len(matches) / len(expected_set)
                      * 100) if expected_set else 0.0

            # Accumulate metrics
            total_latency += latency_ms
            total_precision += precision
            total_recall += recall
            successful_runs += 1

            print(
                f"  ⏱️ Latency: {latency_ms:.0f}ms | 🎯 Precision: {precision:.0f}% | 🔄 Recall: {recall:.0f}% | 📦 Returned: {len(ranked_ids)}")

        except requests.exceptions.RequestException as e:
            print(f"  🚨 Network Error: {e}")
        except json.JSONDecodeError:
            print(f"  🚨 Parse Error: Invalid JSON response.")

    # --- AGGREGATE SUMMARY ---
    if successful_runs > 0:
        avg_latency = total_latency / successful_runs
        avg_precision = total_precision / successful_runs
        avg_recall = total_recall / successful_runs

        print("\n" + "="*50)
        print("📊 BATCH BENCHMARK SUMMARY")
        print("="*50)
        print(
            f"Total Successful Requests: {successful_runs} / {len(TEST_CASES)}")
        print(f"Average Latency:           {avg_latency:.2f} ms")
        print(f"Average Precision:         {avg_precision:.2f}%")
        print(f"Average Recall:            {avg_recall:.2f}%")
        print("="*50)
    else:
        print("\n🚨 All requests failed. Could not compute summary metrics.")


if __name__ == "__main__":
    run_benchmarks()
