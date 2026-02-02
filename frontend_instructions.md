I have completed the development of the Next.js frontend with a ChatGPT-like interface and interactive display of iMessage sources.

Here's how you can run it locally and deploy it to Vercel:

### 1. Run the FastAPI Backend (if not already running)
Navigate to the project root directory and start your FastAPI server:
```bash
uvicorn app:app --reload
```
This should be running, typically on `http://localhost:8000`.

### 2. Run the Next.js Frontend Locally
Navigate into the `frontend` directory and start the Next.js development server:
```bash
cd frontend
npm install
npm run dev
```
The frontend will typically be available at `http://localhost:3000`.

### 3. Deploy to Vercel
1.  **Environment Variable:** In your Vercel project settings, you **must** set the environment variable `NEXT_PUBLIC_FASTAPI_BASE_URL` to the URL of your deployed FastAPI backend (e.g., `https://your-fastapi-app.fly.dev`).
2.  **Vercel CLI (Optional but Recommended):**
    ```bash
    cd frontend
    npm install -g vercel
    vercel login
    vercel deploy
    ```
    Follow the prompts to link your project and deploy. Vercel will automatically detect that it's a Next.js project and configure the build.

The frontend is now ready! Let me know if you want to proceed with any of the tasks from the `README.md`'s to-do list, or if you have further changes.