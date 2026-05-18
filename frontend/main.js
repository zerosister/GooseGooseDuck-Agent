// 使用 ESM 语法导入
import { app, BrowserWindow, ipcMain, screen } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';

// 在 ESM 中需要手动模拟 __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CENTER_NAV_ARG = '--center-navigation';
let mainWindow = null;

const requestNavigationCenter = () => {
    if (mainWindow) {
        if (mainWindow.isMinimized()) mainWindow.restore();
        mainWindow.show();
        mainWindow.webContents.send('center-navigation-bar');
    }
};

const setupTaskbarTasks = () => {
    if (process.platform !== 'win32') return;

    const appPathArg = process.defaultApp ? `"${app.getAppPath()}" ` : '';
    app.setUserTasks([
        {
            program: process.execPath,
            arguments: `${appPathArg}${CENTER_NAV_ARG}`,
            title: '导航栏居中',
            description: '将悬浮导航栏移动到屏幕中间'
        }
    ]);
};

const gotSingleInstanceLock = app.requestSingleInstanceLock();

if (!gotSingleInstanceLock) {
    app.quit();
} else {
    app.on('second-instance', (_event, argv) => {
        if (argv.includes(CENTER_NAV_ARG)) {
            requestNavigationCenter();
        }
    });
}

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
    mainWindow = win;

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

    ipcMain.on('center-navigation-bar', (event, { navCenterX }) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (!win) return;

        const bounds = win.getBounds();
        const display = screen.getDisplayMatching(bounds);
        const workArea = display.workArea;
        const targetNavCenterX = workArea.x + Math.round(workArea.width / 2);
        const maxX = Math.max(workArea.x, workArea.x + workArea.width - bounds.width);
        const nextX = Math.min(
            Math.max(bounds.x + targetNavCenterX - Math.round(navCenterX), workArea.x),
            maxX
        );

        win.setBounds({ x: nextX, y: bounds.y, width: bounds.width, height: bounds.height }, true);
    });

    win.on('closed', () => {
        if (mainWindow === win) mainWindow = null;
    });
}

if (gotSingleInstanceLock) {
    app.whenReady().then(() => {
        setupTaskbarTasks();
        createWindow();

        if (process.argv.includes(CENTER_NAV_ARG)) {
            requestNavigationCenter();
        }
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') app.quit();
    });
}
