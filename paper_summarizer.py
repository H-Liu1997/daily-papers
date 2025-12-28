import os
import json
import datetime
import requests
import pytz
from openai import OpenAI

def sync_to_notion(papers, summaries, database_id, tags=None, date_str=None):
    """Sync papers to Notion database"""
    notion_token = os.getenv('NOTION_API_KEY')
    if not notion_token:
        print("NOTION_API_KEY not set, skipping Notion sync")
        return
    
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    if tags is None:
        tags = []
    
    for paper, summary in zip(papers, summaries):
        if isinstance(summary, dict):
            input_output = summary.get('input_output', '')[:2000]
            problem = summary.get('problem', '')[:2000]
            solution = summary.get('solution', '')[:2000]
        else:
            input_output = str(summary)[:2000]
            problem = ""
            solution = ""
        
        tag_options = [{"name": t} for t in tags] if tags else []
        
        page_data = {
            "parent": {"database_id": database_id},
            "properties": {
                "Name": {"title": [{"text": {"content": paper['title'][:100]}}]},
                "Date": {"date": {"start": date_str}} if date_str else {},
                "Input_Output": {"rich_text": [{"text": {"content": input_output}}]},
                "Problem": {"rich_text": [{"text": {"content": problem}}]},
                "Solution": {"rich_text": [{"text": {"content": solution}}]},
                "ArXiv": {"url": paper['arxiv_url']},
                "HF": {"url": paper['url']},
                "Upvotes": {"number": paper['upvotes']},
                "Tags": {"multi_select": tag_options},
            },
            "children": [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Abstract"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": paper['abstract'][:2000]}}]}
                }
            ]
        }
        
        response = requests.post(
            "https://api.notion.com/v1/pages",
            headers=headers,
            json=page_data
        )
        
        if response.status_code == 200:
            print(f"  -> Synced to Notion: {paper['title'][:50]}...")
        else:
            print(f"  -> Failed to sync: {response.status_code} - {response.text[:200]}")

def get_papers(date_str=None):
    if date_str is None:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        date_str = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d')
    
    url = f"https://huggingface.co/api/daily_papers?date={date_str}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch papers: {response.status_code}")
        return []
    
    papers = response.json()
    valid_papers = []
    for p in papers:
        paper = p.get('paper', {})
        if paper.get('title') and paper.get('summary'):
            valid_papers.append({
                'id': paper.get('id'),
                'title': paper.get('title'),
                'abstract': paper.get('summary'),
                'upvotes': paper.get('upvotes', 0),
                'authors': [a.get('name', '') for a in paper.get('authors', [])],
                'url': f"https://huggingface.co/papers/{paper.get('id')}",
                'arxiv_url': f"https://arxiv.org/abs/{paper.get('id')}"
            })
    
    valid_papers.sort(key=lambda x: x['upvotes'], reverse=True)
    return valid_papers

def summarize_paper(client, paper, model="gpt-4o-mini"):
    prompt = f"""Read this paper abstract. Explain it like you're telling a colleague over coffee - no jargon, no paper-speak.

Title: {paper['title']}

Abstract: {paper['abstract']}

Answer in Chinese, in plain language. Return ONLY valid JSON with these 3 keys:
{{
  "input_output": "What goes in and out? Be concrete with example. 1-2 sentences.",
  "problem": "What couldn't we do before? Why does it matter? 1-2 sentences.", 
  "solution": "What did they build? Core idea simply. 2-3 sentences."
}}

Rules:
- NO academic jargon
- Be specific, give examples
- Keep each field under 200 characters if possible"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    import re
    content = response.choices[0].message.content
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    return {"input_output": content[:200], "problem": "", "solution": ""}

def generate_report(papers, summaries, date_str, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    md_content = f"# HF Daily Papers - {date_str}\n\n"
    md_content += f"Total: {len(papers)} papers\n\n---\n\n"
    
    for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
        md_content += f"## {i}. {paper['title']}\n\n"
        md_content += f"**Upvotes**: {paper['upvotes']} | "
        md_content += f"**Authors**: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n\n"
        md_content += f"[HuggingFace]({paper['url']}) | [arXiv]({paper['arxiv_url']})\n\n"
        
        if isinstance(summary, dict):
            md_content += f"**输入→输出**: {summary.get('input_output', '')}\n\n"
            md_content += f"**问题**: {summary.get('problem', '')}\n\n"
            md_content += f"**方案**: {summary.get('solution', '')}\n\n"
        else:
            md_content += f"{summary}\n\n"
        
        md_content += f"<details>\n<summary>Original Abstract</summary>\n\n{paper['abstract']}\n\n</details>\n\n"
        md_content += "---\n\n"
    
    output_file = os.path.join(output_dir, f"{date_str}_papers.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Report saved to: {output_file}")
    return output_file

def main(date_str=None, top_n=None, keywords=None, notion_db=None, tags=None):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    print(f"Fetching papers for {date_str or 'today'}...")
    papers = get_papers(date_str)
    
    if not papers:
        print("No papers found")
        return
    
    print(f"Found {len(papers)} papers")
    
    if keywords:
        keywords_lower = [k.lower() for k in keywords]
        filtered = []
        for p in papers:
            text = (p['title'] + ' ' + p['abstract']).lower()
            if any(k in text for k in keywords_lower):
                filtered.append(p)
        papers = filtered
        print(f"After keyword filter: {len(papers)} papers")
    
    if top_n and len(papers) > top_n:
        papers = papers[:top_n]
        print(f"Taking top {top_n} papers")
    
    summaries = []
    for i, paper in enumerate(papers, 1):
        print(f"Summarizing {i}/{len(papers)}: {paper['title'][:50]}...")
        summary = summarize_paper(client, paper, model)
        summaries.append(summary)
    
    if date_str is None:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        date_str = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d')
    
    generate_report(papers, summaries, date_str)
    
    if notion_db:
        print("Syncing to Notion...")
        sync_to_notion(papers, summaries, notion_db, tags, date_str)
    
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD)')
    parser.add_argument('--top', type=int, help='Top N papers by upvotes')
    parser.add_argument('--keywords', type=str, nargs='+', help='Filter by keywords')
    parser.add_argument('--notion', type=str, help='Notion database ID to sync to')
    parser.add_argument('--tags', type=str, nargs='+', help='Tags to add in Notion (e.g. "motion generation" "video")')
    args = parser.parse_args()
    
    main(date_str=args.date, top_n=args.top, keywords=args.keywords, notion_db=args.notion, tags=args.tags)

