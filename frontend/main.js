// 使用 ESM 语法导入
import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';

// 在 ESM 中需要手动模拟 __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function createWindow () {
    const win = new BrowserWindow({
        width: 800,   // 初始宽度调大一点
        height: 600,
        alwaysOnTop: true,
        transparent: true,
        frame: false,             // 保持无边框
        resizable: true,          // 允许调整窗口大小
        hasShadow: false,
        webPreferences: {
        // 确保这里的路径指向的是刚才那个 preload.js
        // 使用 path.join(__dirname, 'preload.js') 比较保险
        preload: path.join(__dirname, 'preload.js'), 
        contextIsolation: true,
        nodeIntegration: false,
        }
    });

    // 加载 Vite 开发服务器地址
    win.loadURL('http://localhost:5173');
    
    // 监听来自前端的最小化请求
    ipcMain.on('window-min', () => {
        win.minimize();
    }); 

    ipcMain.on('set-ignore-mouse-events', (event, ignore, options) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win) {
            // 这里的 ignore 应该是 Vue 传过来的 true 或 false
            win.setIgnoreMouseEvents(ignore, options || { forward: true });
        }
    });

    // 窗口缩放监听
    ipcMain.on('resize-window', (event, { width, height }) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win) {
            // setSize 的第三个参数 true 表示允许动画过渡，体验更好
            win.setSize(width, height, true); 
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});