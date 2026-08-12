import "./globals.css";
import { AppStateProvider, ToastProvider } from "./providers";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";

export const metadata = {
  title: "TalentLens — AI Resume Screening",
  description: "Screen candidates against a job description in minutes, with an explainable ranking.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AppStateProvider>
          <ToastProvider>
            <div className="app">
              <Sidebar />
              <main className="main">
                <TopBar />
                {children}
              </main>
            </div>
          </ToastProvider>
        </AppStateProvider>
      </body>
    </html>
  );
}
