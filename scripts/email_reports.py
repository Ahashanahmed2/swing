jobs:
  report-and-mail:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: pip install python-dotenv  # (যদি অন্য কোথাও লাগে) আসলে এখানে লাগছে না

      - name: Generate PDF report
        run: python generate_report.py   # ফোল্ডার output/ai_signal/pdfs বানাবে

      - name: Send email
        env:
          EMAIL_USER: ${{ secrets.MAIL_USER }}
          EMAIL_PASS: ${{ secrets.MAIL_PASS }}
          EMAIL_TO:   ${{ secrets.MAIL_TO  }}
        run: python scripts/send_email.py
