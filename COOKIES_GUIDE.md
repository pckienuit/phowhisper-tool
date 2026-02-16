# How to Fix "Sign in to confirm you’re not a bot" Error

The `cookies.txt` file is missing in your project folder (`d:\phowhisper-tool`). You **MUST** create this file to bypass YouTube's bot detection.

## 🟢 Step-by-Step Instructions

1.  **Install a Cookie Exporter Extension**:
    *   **Chrome / Thorium / Brave**: Install [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiesha/cclelndahbckbenkjhflpcmgdldjajaal)
    *   **Firefox**: Install [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2.  **Export Cookies**:
    1.  Open your browser (Thorium/Chrome) and go to **[YouTube.com](https://www.youtube.com)**.
    2.  Make sure you remain **Logged In**.
    3.  Click the **"Get cookies.txt LOCALLY"** extension icon in your toolbar.
    4.  Click **"Export"** or **"Download"**.

3.  **Save the File**:
    *   Rename the downloaded file to: `cookies.txt`
    *   **Move/Save it to**: `d:\phowhisper-tool\cookies.txt`

4.  **Verify**:
    *   You should see `d:\phowhisper-tool\cookies.txt` alongside `phowhisper.py`.

5.  **Run the Tool Again**:
    ```bash
    python phowhisper.py https://www.youtube.com/watch?v=N3vHJcHBS-w
    ```
