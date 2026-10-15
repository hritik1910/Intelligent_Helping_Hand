from src.assistant import explain_results

result = explain_results(
    question="What does first-pass yield mean?",
    verified_results={
        "example_units_produced": 1000,
        "example_units_passed_first_pass": 960,
        "example_fpy_pct": 96.0
    }
)

print(result)