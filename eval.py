"""
eval.py

Simple enterprise RAG retrieval evaluation.

This tests whether the expected policy section appears in
the retrieved Top-K results.

Run:

    python eval.py
"""

from rag import create_rag


# ============================================================
# TEST DATASET
# ============================================================

TEST_CASES = [

    {
        "question":
            "What happens to my outstanding loan if I resign?",

        "expected_section":
            "Loans & Advances",
    },

    {
        "question":
            "Can the company pay for my professional certification?",

        "expected_section":
            "Certification & Professional Membership",
    },

    {
        "question":
            "What happens when an employee resigns?",

        "expected_section":
            "Separation",
    },

    {
        "question":
            "What rules apply to international business travel?",

        "expected_section":
            "International Travel",
    },

    {
        "question":
            "What health insurance benefits are available?",

        "expected_section":
            "Health & Insurance",
    },

    {
        "question":
            "How many hours should employees normally work?",

        "expected_section":
            "Working Hours",
    },

    {
        "question":
            "What is the company's leave policy?",

        "expected_section":
            "Leave",
    },

    {
        "question":
            "Can I work flexible hours?",

        "expected_section":
            "Flexible Working",
    },

    {
        "question":
            "What is the policy for social media use?",

        "expected_section":
            "Social Media",
    },

    {
        "question":
            "Can I attend an external professional conference?",

        "expected_section":
            "Seminars & Conferences",
    },

]


TOP_K = 10


# ============================================================
# EVALUATION
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "ENTERPRISE POLICY RAG EVALUATION"
    )

    print("=" * 80)


    rag = create_rag()


    passed = 0

    failed = 0


    failures = []


    for number, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        question = (
            test["question"]
        )

        expected = (
            test["expected_section"]
        )


        chunks = (
            rag.retriever.retrieve(

                question=question,

                top_k=TOP_K,
            )
        )


        retrieved_sections = []


        for chunk in chunks:

            section = " > ".join(

                heading

                for heading in [

                    chunk.heading_1,
                    chunk.heading_2,
                    chunk.heading_3,

                ]

                if heading
            )


            retrieved_sections.append(
                section
            )


        found = any(

            expected.lower()
            in section.lower()

            for section
            in retrieved_sections
        )


        if found:

            status = "PASS"

            passed += 1


        else:

            status = "FAIL"

            failed += 1


            failures.append({

                "question":
                    question,

                "expected":
                    expected,

                "retrieved":
                    retrieved_sections,
            })


        print(
            f"{number:02d}. "
            f"{status:<4} | "
            f"{question}"
        )


        print(
            f"    Expected: "
            f"{expected}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        TEST_CASES
    )


    recall = (
        passed / total
        if total
        else 0
    )


    print()
    print("=" * 80)

    print(
        "RESULT"
    )

    print("=" * 80)


    print(
        f"Total tests:       {total}"
    )

    print(
        f"Passed:            {passed}"
    )

    print(
        f"Failed:            {failed}"
    )

    print(
        f"Retrieval Recall@{TOP_K}: "
        f"{recall:.1%}"
    )


    # ========================================================
    # FAILURES
    # ========================================================

    if failures:

        print()
        print("=" * 80)

        print(
            "FAILED RETRIEVAL CASES"
        )

        print("=" * 80)


        for failure in failures:

            print()

            print(
                "Question:",
                failure["question"]
            )

            print(
                "Expected:",
                failure["expected"]
            )

            print(
                "Retrieved:"
            )


            for section in (
                failure["retrieved"]
            ):

                print(
                    "  -",
                    section or "Unknown"
                )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()