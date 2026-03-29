"use client";
import { useState } from "react";
import LoginModal from "../components/loginModal";
import ChatBot from "../components/ChatBox";
import { useAuth } from "../context/AuthContext";

export default function Home() {
  const { user } = useAuth();
  const [showModal, setShowModal] = useState(false);

  return (
    <div>
      {!user ? (
        <div className="landing">
          <h1>FinBot 💼</h1>
          <button onClick={() => setShowModal(true)}>
            Login
          </button>

          {showModal && (
            <LoginModal onClose={() => setShowModal(false)} />
          )}
        </div>
      ) : (
        <ChatBot />
      )}
    </div>
  );
}