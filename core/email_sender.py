import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompts import EMAIL_PROMPT

load_dotenv()

def send_emails_to_candidates(candidates: list, query_context: str, llm):
    sent_count = 0
    for cand in candidates:
        if not cand.get("email"):
            continue

        # Generate personalized email with Gemini
        chain = EMAIL_PROMPT | llm
        email_content = chain.invoke({
            "query_context": query_context,
            "candidate_name": cand["name"],
            "primary_field": cand["primary_field"],
            "experience_years": cand["experience_years"]
        })

        # Parse subject & body (simple split)
        lines = email_content.content.split("Body:", 1)
        subject = lines[0].replace("Subject:", "").strip()
        body = lines[1].strip() if len(lines) > 1 else email_content.content

        # Send
        try:
            msg = MIMEMultipart()
            msg["From"] = os.getenv("SENDER_EMAIL")
            msg["To"] = cand["email"]
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
                server.starttls()
                server.login(os.getenv("SENDER_EMAIL"), os.getenv("SENDER_PASSWORD"))
                server.send_message(msg)
            sent_count += 1
        except Exception as e:
            print(f"Failed to send to {cand['email']}: {e}")

    return sent_count