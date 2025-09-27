import os
import json
import requests
import ollama

def get_pr_diff(repo, pr_number, token):
    """Fetches the diff of a pull request from the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"token {token}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def analyze_code_with_ollama(diff):
    """Analyzes the code diff using a local Ollama model."""
    prompt = f"""
You are an expert, automated code reviewer. Your task is to analyze the following code changes from a pull request and provide concise, constructive feedback.

Focus on the following:
- Potential bugs or logical errors.
- Security vulnerabilities.
- Adherence to best practices.
- Performance issues.
- Code clarity and maintainability.

Do not comment on code style unless it severely impacts readability. Provide specific examples from the code where possible. Structure your feedback clearly. If there are no issues, simply state "No issues found."

Here is the code diff:
```diff
{diff}
```
"""
    try:
        client = ollama.Client()
        response = client.chat(
            model='gemma:2b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error analyzing code with Ollama: {e}"

def post_review_comment(repo, pr_number, comment, token):
    """Posts a comment on the pull request."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
    }
    data = {"body": f"### AI Code Review\n\n{comment}"}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def main():
    """Main function to run the code review process."""
    # --- 1. Get environment variables from GitHub Actions ---
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    event_path = os.getenv("GITHUB_EVENT_PATH")

    if not all([token, repo, event_path]):
        print("Error: Missing required environment variables.")
        return

    # --- 2. Load PR data from the event payload ---
    try:
        with open(event_path, 'r') as f:
            event_data = json.load(f)
        pr_number = event_data["pull_request"]["number"]
    except (IOError, KeyError) as e:
        print(f"Error reading event payload or finding PR number: {e}")
        return

    print(f"Starting review for PR #{pr_number} in repository {repo}...")

    # --- 3. Fetch the PR diff ---
    try:
        diff = get_pr_diff(repo, pr_number, token)
        if not diff:
            print("No diff found. Exiting.")
            return
    except requests.HTTPError as e:
        print(f"Error fetching PR diff: {e}")
        return

    # --- 4. Analyze the diff with the local LLM ---
    print("Analyzing code with local LLM...")
    review = analyze_code_with_ollama(diff)
    print("Analysis complete.")

    # --- 5. Post the review as a comment on the PR ---
    try:
        print("Posting review comment to GitHub...")
        post_review_comment(repo, pr_number, review, token)
        print("Successfully posted review comment.")
    except requests.HTTPError as e:
        print(f"Error posting comment to GitHub: {e}")
        print(f"Response body: {e.response.text}")

if __name__ == "__main__":
    main()