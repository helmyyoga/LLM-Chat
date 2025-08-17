css = '''
<style>
.chat-message {
    padding: 1rem 1.5rem; 
    border-radius: 1rem; 
    margin: 1rem 0; 
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeIn 0.4s ease-in-out;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: transform 0.2s ease;
}
.chat-message:hover {
    transform: translateY(-2px);
}

.chat-message.user {
    background: linear-gradient(135deg, #2b313e, #3c4455);
    border-left: 5px solid #4ea1ff;
}
.chat-message.bot {
    background: linear-gradient(135deg, #475063, #5a6479);
    border-left: 5px solid #ffb84d;
}

.chat-message .avatar {
  width: 50px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
.chat-message .avatar img {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
  background: #fff;
  padding: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.chat-message .message {
  flex-grow: 1;
  color: #fff;
  font-size: 0.95rem;
  line-height: 1.5;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" alt="bot avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/2202/2202112.png" alt="user avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
