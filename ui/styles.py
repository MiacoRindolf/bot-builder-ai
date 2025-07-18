MAIN_STYLES = '''
<style>
body {
    background: #181c2a;
    color: #f4f4f4;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.metric-card {
    background: #232946;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #232946;
}

/* Right Sidebar Chat */
.right-sidebar-chat {
    background: #232946;
    border-radius: 12px;
    padding: 1rem;
    height: 600px;
    display: flex;
    flex-direction: column;
}
.chat-messages-container {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 1rem;
    padding: 0.5rem;
    background: #181c2a;
    border-radius: 8px;
    max-height: 400px;
}
.chat-input-container {
    margin-top: auto;
}

/* Chat Interface */
.chat-container {
    background: #232946;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}
.chat-messages {
    height: 400px;
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background: #181c2a;
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid #393e5c;
}
.user-message {
    background: #5a6fd8;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0 0.8rem 3rem;
    text-align: right;
    max-width: 70%;
    margin-left: auto;
    word-wrap: break-word;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.ai-message {
    background: #393e5c;
    color: #f4f4f4;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 3rem 0.8rem 0;
    text-align: left;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.chat-input-row {
    display: flex;
    align-items: center;
    padding: 1rem;
    background: #232946;
    border-radius: 0 0 16px 16px;
    gap: 0.5rem;
}
.stTextInput > div > div > input {
    background: #181c2a;
    color: #f4f4f4;
    border-radius: 8px;
    border: 1px solid #393e5c;
    padding: 0.5rem 1rem;
}
.stButton > button {
    background: #393e5c;
    color: #fff;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #232946;
}

/* Completely hide Streamlit form styling */
.stForm {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.stForm > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.stForm > div > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
/* Force input area to match chat container width */
.stForm {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Hide any potential sender labels */
.user-message strong,
.ai-message strong {
    display: none !important;
}

/* Ensure proper scrolling */
div[style*="overflow-y:auto"] {
    scrollbar-width: thin;
    scrollbar-color: #393e5c #181c2a;
}

div[style*="overflow-y:auto"]::-webkit-scrollbar {
    width: 8px;
}

div[style*="overflow-y:auto"]::-webkit-scrollbar-track {
    background: #181c2a;
}

div[style*="overflow-y:auto"]::-webkit-scrollbar-thumb {
    background: #393e5c;
    border-radius: 4px;
}

div[style*="overflow-y:auto"]::-webkit-scrollbar-thumb:hover {
    background: #4a4f6a;
}
</style>
''' 