import { Message } from "@/types";
import styles from "./MessageBubble.module.css";

export default function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === "user";

    return (
        <div className={`${styles.messageRow} ${isUser ? styles.user : styles.assistant}`}>
            <div className={styles.messageBubble}>{message.content}</div>
        </div>
    );
}