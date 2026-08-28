import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata = {
  title: "Adversarial ML Security Playground",
  description: "Attack, defend, and evaluate the robustness of ML models.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="font-sans bg-bg text-ink">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 px-8 py-8 max-w-6xl">{children}</main>
        </div>
      </body>
    </html>
  );
}
