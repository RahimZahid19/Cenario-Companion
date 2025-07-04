from playwright.sync_api import sync_playwright

def save_gmail_auth():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        page = context.new_page()
        page.goto("https://accounts.google.com")

        print("Please log in manually to your Gmail account in the browser window...")

        input("Press ENTER after login is complete and you can see your inbox or profile page.")

        # Save login state to auth.json
        context.storage_state(path="auth.json")
        print("Login session saved to auth.json")

        browser.close()

if __name__ == "__main__":
    save_gmail_auth()
