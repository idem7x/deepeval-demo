import { redirect } from "next/navigation";

export default function Home() {
  // Root always lands on the chat — that is the main demo surface.
  redirect("/chat");
}
