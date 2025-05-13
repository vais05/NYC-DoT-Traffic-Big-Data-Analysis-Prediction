import os
import json
import re
import base64
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from IPython.display import display, HTML

def read_notebook(notebook_path):
    """Read and parse a Jupyter notebook file"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading notebook {notebook_path}: {e}")
        return None

def get_markdown_content(cell):
    """Convert markdown cell to HTML"""
    try:
        # In a real implementation, you might want to use a proper Markdown renderer
        # For now, we're just returning the source as-is to be rendered in the HTML page
        return "\n".join(cell["source"])
    except Exception as e:
        print(f"Error processing markdown: {e}")
        return ""

def get_code_content(cell):
    """Get code cell content"""
    try:
        return "\n".join(cell["source"])
    except Exception as e:
        print(f"Error processing code: {e}")
        return ""

def get_output_content(cell):
    """Extract output from a code cell"""
    outputs = []
    
    if "outputs" not in cell:
        return ""
    
    for output in cell["outputs"]:
        if output["output_type"] == "stream":
            outputs.append("".join(output.get("text", [])))
        
        elif output["output_type"] == "execute_result":
            if "data" in output:
                if "text/html" in output["data"]:
                    html_content = "".join(output["data"]["text/html"])
                    outputs.append(html_content)
                elif "text/plain" in output["data"]:
                    outputs.append("".join(output["data"]["text/plain"]))
        
        elif output["output_type"] == "display_data":
            if "data" in output:
                if "text/html" in output["data"]:
                    html_content = "".join(output["data"]["text/html"])
                    outputs.append(html_content)
                elif "image/png" in output["data"]:
                    image_data = output["data"]["image/png"]
                    if isinstance(image_data, list):
                        image_data = "".join(image_data)
                    img_tag = f'<img src="data:image/png;base64,{image_data}" />'
                    outputs.append(img_tag)
                elif "text/plain" in output["data"]:
                    outputs.append("".join(output["data"]["text/plain"]))
        
        elif output["output_type"] == "error":
            error_content = "\n".join([f"{output.get('ename', '')}: {output.get('evalue', '')}"] + 
                                    output.get("traceback", []))
            outputs.append(f'<pre class="error">{error_content}</pre>')
    
    return "\n".join(outputs)

def create_page_content(notebook_data, notebook_name):
    """Create HTML content for a notebook"""
    title = notebook_name.replace('_', ' ').replace('.ipynb', '')
    cells_html = []
    
    # Try to extract a description from the first markdown cell
    description = "Notebook for NYC Traffic Analysis"
    for cell in notebook_data.get("cells", []):
        if cell["cell_type"] == "markdown":
            markdown_content = "".join(cell.get("source", []))
            if markdown_content.strip():
                # Extract the first paragraph or heading as description
                first_line = markdown_content.strip().split('\n')[0]
                if first_line.startswith('#'):
                    # Remove markdown heading syntax
                    description = re.sub(r'^#+\s*', '', first_line)
                else:
                    description = first_line[:100] + ('...' if len(first_line) > 100 else '')
                break
    
    for cell in notebook_data.get("cells", []):
        if cell["cell_type"] == "markdown":
            content = get_markdown_content(cell)
            cells_html.append(f'''
            <div class="cell markdown-cell">
                {content}
            </div>
            ''')
        
        elif cell["cell_type"] == "code":
            code_content = get_code_content(cell)
            output_content = get_output_content(cell)
            
            if code_content.strip():
                cells_html.append(f'''
                <div class="cell">
                    <div class="code-cell">
                        <pre><code>{code_content}</code></pre>
                    </div>
                    {f'<div class="output-cell">{output_content}</div>' if output_content else ''}
                </div>
                ''')
    
    return {
        "title": title,
        "description": description,
        "content": "\n".join(cells_html)
    }

def get_notebook_info(notebook_path):
    """Extract basic info from a notebook without full processing"""
    notebook_name = os.path.basename(notebook_path)
    notebook_data = read_notebook(notebook_path)
    
    title = notebook_name.replace('_', ' ').replace('.ipynb', '')
    description = "Notebook for NYC Traffic Analysis"
    
    if notebook_data:
        for cell in notebook_data.get("cells", []):
            if cell["cell_type"] == "markdown":
                markdown_content = "".join(cell.get("source", []))
                if markdown_content.strip():
                    first_line = markdown_content.strip().split('\n')[0]
                    if first_line.startswith('#'):
                        description = re.sub(r'^#+\s*', '', first_line)
                    else:
                        description = first_line[:100] + ('...' if len(first_line) > 100 else '')
                    break
    
    return {
        "path": notebook_path,
        "name": notebook_name,
        "title": title,
        "description": description,
        "id": notebook_name.split('_')[0]
    }

def generate_html_file(notebook_path, output_dir, all_notebooks):
    """Generate HTML file from a Jupyter notebook"""
    notebook_data = read_notebook(notebook_path)
    if not notebook_data:
        return False
    
    notebook_name = os.path.basename(notebook_path)
    output_filename = notebook_name.replace('.ipynb', '.html')
    output_path = os.path.join(output_dir, output_filename)
    
    page_content = create_page_content(notebook_data, notebook_name)
    
    # Generate navigation links
    nav_links = []
    for nb in all_notebooks:
        active = nb["name"] == notebook_name
        html_file = nb["name"].replace('.ipynb', '.html')
        nav_links.append(f'<a href="{html_file}" id="nav-{nb["id"]}" class="{" active" if active else ""}">{nb["title"]}</a>')
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Traffic Analysis - {page_content["title"]}</title>
    <style>
        :root {{
            --primary-color: #1e3a8a;
            --secondary-color: #3b82f6;
            --accent-color: #f59e0b;
            --text-color: #374151;
            --background-color: #f9fafb;
            --code-bg: #f3f4f6;
            --border-color: #e5e7eb;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        
        .project-description {{
            margin-top: 0.5rem;
            font-size: 1rem;
            opacity: 0.9;
        }}
        
        nav {{
            background-color: white;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
            overflow-x: auto;
            white-space: nowrap;
        }}
        
        .nav-links {{
            display: flex;
            gap: 1rem;
            padding: 0.75rem 0;
        }}
        
        .nav-links a {{
            text-decoration: none;
            color: var(--text-color);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .nav-links a:hover {{
            background-color: var(--code-bg);
            color: var(--secondary-color);
        }}
        
        .nav-links a.active {{
            background-color: var(--secondary-color);
            color: white;
        }}
        
        main {{
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}
        
        .notebook-container {{
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .notebook-header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        
        h2 {{
            color: var(--primary-color);
            margin-top: 0;
        }}
        
        .notebook-description {{
            color: var(--text-color);
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }}
        
        .cell {{
            margin-bottom: 2rem;
            border-radius: 0.25rem;
            overflow: hidden;
        }}
        
        .code-cell {{
            background-color: var(--code-bg);
            border-left: 4px solid var(--secondary-color);
            padding: 1rem;
            margin-bottom: 1rem;
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
        }}
        
        .markdown-cell {{
            background-color: white;
            padding: 0.5rem 1rem;
            border-left: 4px solid var(--accent-color);
        }}
        
        .output-cell {{
            background-color: white;
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 0.25rem;
            margin-top: 0.5rem;
            overflow-x: auto;
        }}
        
        .output-cell img {{
            max-width: 100%;
            height: auto;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }}
        
        .output-text {{
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        
        th, td {{
            border: 1px solid var(--border-color);
            padding: 0.5rem;
            text-align: left;
        }}
        
        th {{
            background-color: var(--code-bg);
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        
        footer {{
            background-color: var(--primary-color);
            color: white;
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
        }}
        
        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        @media (max-width: 768px) {{
            header, main {{
                padding: 1rem;
            }}
            
            .header-content {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .nav-links {{
                overflow-x: auto;
            }}
        }}
        
        /* Specialized styling for markdown */
        .markdown-cell h1, .markdown-cell h2, .markdown-cell h3, 
        .markdown-cell h4, .markdown-cell h5, .markdown-cell h6 {{
            color: var(--primary-color);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        .markdown-cell h1 {{ font-size: 1.8rem; }}
        .markdown-cell h2 {{ font-size: 1.5rem; }}
        .markdown-cell h3 {{ font-size: 1.3rem; }}
        
        .markdown-cell p {{
            margin-bottom: 1rem;
        }}
        
        .markdown-cell ul, .markdown-cell ol {{
            margin-bottom: 1rem;
            padding-left: 2rem;
        }}
        
        .markdown-cell a {{
            color: var(--secondary-color);
            text-decoration: none;
        }}
        
        .markdown-cell a:hover {{
            text-decoration: underline;
        }}
        
        .markdown-cell code {{
            background-color: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        .markdown-cell pre {{
            background-color: var(--code-bg);
            padding: 1rem;
            border-radius: 0.25rem;
            overflow-x: auto;
            margin-bottom: 1rem;
        }}
        
        .markdown-cell pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        .markdown-cell blockquote {{
            border-left: 4px solid var(--secondary-color);
            padding-left: 1rem;
            margin-left: 0;
            color: #4b5563;
        }}
        
        .error {{
            color: #ef4444;
            background-color: #fee2e2;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border-left: 4px solid #ef4444;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div>
                <h1>NYC Traffic Analysis</h1>
                <div class="project-description">Analysis of NYC Yellow Taxi Trip Data using Machine Learning</div>
            </div>
        </div>
    </header>
    
    <nav>
        <div class="nav-container">
            <div class="nav-links">
                {' '.join(nav_links)}
            </div>
        </div>
    </nav>
    
    <main>
        <div class="notebook-container">
            <div class="notebook-header">
                <h2>{page_content["title"]}</h2>
                <div class="notebook-description">{page_content["description"]}</div>
            </div>
            
            {page_content["content"]}
        </div>
    </main>
    
    <footer>
        <div class="footer-content">
            <p>NYC Traffic Analysis Project &copy; 2025</p>
        </div>
    </footer>
</body>
</html>'''

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated HTML file: {output_path}")
    return True

def generate_index_file(output_dir, all_notebooks):
    """Generate index.html file with links to all notebooks"""
    # Create HTML content
    notebook_links = []
    for nb in all_notebooks:
        html_file = nb["name"].replace('.ipynb', '.html')
        notebook_links.append(f'''
        <div class="notebook-card">
            <h3><a href="{html_file}">{nb["title"]}</a></h3>
            <p>{nb["description"]}</p>
        </div>
        ''')
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Traffic Analysis - Project Dashboard</title>
    <style>
        :root {{
            --primary-color: #1e3a8a;
            --secondary-color: #3b82f6;
            --accent-color: #f59e0b;
            --text-color: #374151;
            --background-color: #f9fafb;
            --code-bg: #f3f4f6;
            --border-color: #e5e7eb;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        
        .project-description {{
            margin-top: 0.5rem;
            font-size: 1rem;
            opacity: 0.9;
        }}
        
        main {{
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}
        
        .dashboard-container {{
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            padding: 2rem;
        }}
        
        .dashboard-header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        
        h2 {{
            color: var(--primary-color);
            margin-top: 0;
        }}
        
        .notebook-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .notebook-card {{
            background-color: white;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        
        .notebook-card:hover {{
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }}
        
        .notebook-card h3 {{
            margin-top: 0;
            color: var(--primary-color);
        }}
        
        .notebook-card a {{
            text-decoration: none;
            color: var(--primary-color);
        }}
        
        .notebook-card a:hover {{
            color: var(--secondary-color);
        }}
        
        .notebook-card p {{
            color: var(--text-color);
            margin-bottom: 0;
        }}
        
        footer {{
            background-color: var(--primary-color);
            color: white;
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
        }}
        
        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        @media (max-width: 768px) {{
            header, main {{
                padding: 1rem;
            }}
            
            .header-content {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .notebook-cards {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div>
                <h1>NYC Traffic Analysis</h1>
                <div class="project-description">Analysis of NYC Yellow Taxi Trip Data using Machine Learning</div>
            </div>
        </div>
    </header>
    
    <main>
        <div class="dashboard-container">
            <div class="dashboard-header">
                <h2>Project Dashboard</h2>
                <p>Welcome to the NYC Traffic Analysis project. Explore our analysis workflow through the notebooks below.</p>
            </div>
            
            <div class="notebook-cards">
                {' '.join(notebook_links)}
            </div>
        </div>
    </main>
    
    <footer>
        <div class="footer-content">
            <p>NYC Traffic Analysis Project &copy; 2025</p>
        </div>
    </footer>
</body>
</html>'''

    # Write HTML file
    output_path = os.path.join(output_dir, 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated index file: {output_path}")
    return True

def process_notebooks(notebooks_dir, output_dir):
    """Process all notebooks in the specified directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all notebooks
    notebooks = []
    for filename in sorted(os.listdir(notebooks_dir)):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(notebooks_dir, filename)
            notebooks.append(get_notebook_info(notebook_path))
    
    # Generate HTML files for each notebook
    for notebook in notebooks:
        generate_html_file(notebook["path"], output_dir, notebooks)
    
    # Generate index file
    generate_index_file(output_dir, notebooks)
    
    print(f"Generated HTML files for {len(notebooks)} notebooks in {output_dir}")
    print("Done! Open index.html to view the dashboard.")

if __name__ == "__main__":
    # Set default directories
    notebooks_dir = "./notebooks"
    output_dir = "./html_output"
    
    # Check if directories are provided as command line arguments
    if len(sys.argv) > 1:
        notebooks_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Process notebooks
    process_notebooks(notebooks_dir, output_dir)