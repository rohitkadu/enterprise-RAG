from rag import create_rag


def print_sources(sources):

    print("\nSOURCES")
    print("=" * 70)


    for source in sources:

        print(
            f"\nSource {source['number']}"
        )

        print(
            f"Similarity: {source['score']}"
        )

        print(
            f"Document: {source['source']}"
        )

        print(
            f"Section: {source['section']}"
        )

        print("-" * 70)


def main():

    print("=" * 70)

    print(
        "ACME EMPLOYEE POLICY ASSISTANT"
    )

    print("=" * 70)

    print(
        "Type 'exit' to quit."
    )


    print(
        "\nConnecting to Gemini and Qdrant..."
    )


    rag = create_rag()


    print(
        "Ready."
    )


    while True:

        print("\n")

        question = input(
            "You: "
        ).strip()


        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "Goodbye."
            )

            break


        if not question:

            continue


        try:

            result = rag.ask(
                question
            )


            print("\nAssistant:\n")

            print(
                result["answer"]
            )


            print_sources(
                result["sources"]
            )


        except Exception as error:

            print(
                "\nERROR:"
            )

            print(error)


if __name__ == "__main__":

    main()